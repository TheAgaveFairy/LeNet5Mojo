from std.subprocess import run  # var current = run("date")
import std.os as os
from std.reflection import reflect
from constants import act_fn


@fieldwise_init
struct LogFormat(Copyable, Movable):
    var value: Int  # enum esque

    comptime CSV = LogFormat(0)
    comptime JSON = LogFormat(1)  # TODO: implement

    def __eq__(self, other: LogFormat) -> Bool:
        return self.value == other.value

    def __ne__(self, other: LogFormat) -> Bool:
        return self.value != other.value


trait LogEntry:
    def toCSV(self) -> String:
        ...

    @staticmethod
    def getHeaders() -> String:
        ...

    # TODO: add JSON


struct InferenceResult(LogEntry):
    var timestamp: String
    var device: String
    var elapsed_ns: UInt  # perf_counter_ns() -> UInt
    var correct: Int
    var test_size: Int
    var stream_batch_size: Int
    var num_streams: Int
    var ftype: DType
    var activation_fn: String

    def __init__(
        out self,
        device: String,
        elapsed_ns: UInt,
        correct: Int,
        test_size: Int,
        stream_batch_size: Int,
        num_streams: Int,
        ftype: DType,
    ):
        try:
            self.timestamp = run("date")  # subprocess
        except e:
            print("InferenceResult timestamp error:", e)
            self.timestamp = "TIMESTAMP FAILED"

        self.device = device
        self.elapsed_ns = elapsed_ns
        self.correct = correct
        self.test_size = test_size
        self.stream_batch_size = stream_batch_size
        self.num_streams = num_streams
        self.ftype = ftype
        self.activation_fn = reflect[act_fn].base_name()

    def toCSV(self) -> String:
        var datatype_string = "Float32"
        if self.ftype == DType.float64:
            datatype_string = "Float64"
        return (
            self.timestamp
            + ","
            + self.device
            + ","
            + String(self.elapsed_ns)
            + ","
            + String(self.correct)
            + ","
            + String(self.test_size)
            + ","
            + String(self.stream_batch_size)
            + ","
            + String(self.num_streams)
            + ","
            + datatype_string
            + ","
            + self.activation_fn
        )

    @staticmethod
    def getHeaders() -> String:
        comptime r = reflect[Self]
        comptime names = r.field_names()
        var header = String("")
        comptime for i in range(r.field_count()):
            if i > 0:
                header += ","
            header += String(names[i])
        return header


struct TrainingResult(LogEntry):
    var timestamp: String
    var device: String
    var epoch: Int
    var elapsed_ns: UInt
    var correct: Int
    var test_size: Int  # kinda just gonna be the batch_size since CPU only for now # TODO:
    var loss: Float32
    var learning_rate: Float32
    var ftype: DType
    var activation_fn: String

    def __init__(
        out self,
        device: String,
        epoch: Int,
        elapsed_ns: UInt,
        correct: Int,
        test_size: Int,
        loss: Float32,
        learning_rate: Float32,
        ftype: DType,
    ):
        try:
            self.timestamp = run("date")  # subprocess
        except e:
            print("TrainingResult timestamp error:", e)
            self.timestamp = "TIMESTAMP FAILED"

        self.device = device
        self.epoch = epoch
        self.elapsed_ns = elapsed_ns
        self.correct = correct
        self.test_size = test_size
        self.loss = loss
        self.learning_rate = learning_rate
        self.ftype = ftype
        self.activation_fn = reflect[act_fn].base_name()

    def toCSV(self) -> String:
        var datatype_string = "Float32"
        if self.ftype == DType.float64:
            datatype_string = "Float64"
        return (
            self.timestamp
            + ","
            + self.device
            + ","
            + String(self.epoch)
            + ","
            + String(self.elapsed_ns)
            + ","
            + String(self.correct)
            + ","
            + String(self.test_size)
            + ","
            + String(self.loss)
            + ","
            + String(self.learning_rate)
            + ","
            + datatype_string
            + ","
            + self.activation_fn
        )

    @staticmethod
    def getHeaders() -> String:
        comptime r = reflect[Self]
        comptime names = r.field_names()
        var header = String("")
        comptime for i in range(r.field_count()):
            if i > 0:
                header += ","
            header += String(names[i])
        return header


trait MyLogger:
    def logInferenceResult(
        mut self,
        device: String,
        elapsed_ns: UInt,
        correct: Int,
        test_size: Int,
        stream_batch_size: Int,
        num_streams: Int,
        ftype: DType,
    ) raises -> None:
        ...

    def logTrainingEpoch(
        mut self,
        device: String,
        epoch: Int,
        elapsed_ns: UInt,
        correct: Int,
        test_size: Int,
        loss: Float32,
        learning_rate: Float32,
        ftype: DType,
    ) raises -> None:
        ...


comptime LeNet5Logger = MyLogger & Copyable & Movable


@fieldwise_init
struct ResultLogger(LeNet5Logger):
    var output_path: String
    var format_type: LogFormat
    var headers_written: Bool  # TODO: make this better

    def __init__(
        out self, output_path: String, format_type: LogFormat = LogFormat.CSV
    ):
        self.output_path = output_path
        self.format_type = format_type.copy()
        try:
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    if f.read(10) == "timestamp,":
                        self.headers_written = True
                    else:
                        self.headers_written = False
            else:
                self.headers_written = False
        except e:
            print("ResultLogger init error:", e)
            self.headers_written = False

    def logInferenceResult(
        mut self,
        device: String,
        elapsed_ns: UInt,
        correct: Int,
        test_size: Int,
        stream_batch_size: Int,
        num_streams: Int,
        ftype: DType,
    ) raises -> None:
        var result = InferenceResult(
            device,
            elapsed_ns,
            correct,
            test_size,
            stream_batch_size,
            num_streams,
            ftype,
        )
        self._writeResult(result)

    def logTrainingEpoch(
        mut self,
        device: String,
        epoch: Int,
        elapsed_ns: UInt,
        correct: Int,
        test_size: Int,
        loss: Float32,
        learning_rate: Float32,
        ftype: DType,
    ) raises -> None:
        var result = TrainingResult(
            device,
            epoch,
            elapsed_ns,
            correct,
            test_size,
            loss,
            learning_rate,
            ftype,
        )
        self._writeResult(result)

    def _writeResult[T: LogEntry](mut self, result: T) raises -> None:
        var content: String = ""

        if not self.headers_written:
            if self.format_type == materialize[LogFormat.CSV]():
                content += result.getHeaders() + "\n"
            else:
                content += "INVALID HEADER\n"

            self.headers_written = True

        if self.format_type == materialize[LogFormat.CSV]():
            content += result.toCSV() + "\n"
        else:
            content += "INVALID CONTENT\n"

        with open(self.output_path, "a") as file:
            file.write(content)


@fieldwise_init
struct MultiFileLogger(LeNet5Logger):
    var base_path: String
    var format_type: LogFormat
    var inference_logger: ResultLogger
    var training_logger: ResultLogger

    def __init__(
        out self,
        base_path: String = "results/",
        inference_name: String = "inference",
        training_name: String = "training",
        *,
        format: LogFormat = LogFormat.CSV,
    ):
        self.base_path = base_path
        self.format_type = format.copy()

        var ext = ".csv" if format == materialize[LogFormat.CSV]() else (
            ".json" if format == materialize[LogFormat.JSON]() else ".tsv"
        )

        self.inference_logger = ResultLogger(
            base_path + inference_name + ext, format
        )
        self.training_logger = ResultLogger(
            base_path + training_name + ext, format
        )

    def logInferenceResult(
        mut self,
        device: String,
        elapsed_ns: UInt,
        correct: Int,
        test_size: Int,
        stream_batch_size: Int,
        num_streams: Int,
        ftype: DType,
    ) raises -> None:
        self.inference_logger.logInferenceResult(
            device,
            elapsed_ns,
            correct,
            test_size,
            stream_batch_size,
            num_streams,
            ftype,
        )

    def logTrainingEpoch(
        mut self,
        device: String,
        epoch: Int,
        elapsed_ns: UInt,
        correct: Int,
        test_size: Int,
        loss: Float32,
        learning_rate: Float32,
        ftype: DType,
    ) raises -> None:
        self.training_logger.logTrainingEpoch(
            device,
            epoch,
            elapsed_ns,
            correct,
            test_size,
            loss,
            learning_rate,
            ftype,
        )


def main() raises:
    comptime output_path = "results/"
    var logger = MultiFileLogger(output_path)

    logger.logInferenceResult("RTX6069", 420, 99, 100, 10, 4, DType.float64)
    logger.logInferenceResult("GTX730", 9001, 98, 100, 10, 1, DType.float32)

    logger.logTrainingEpoch(
        "7600X", 1, 69, 11, 100, 0.1337, 0.10, DType.float32
    )
    print("Results logged at", output_path)
