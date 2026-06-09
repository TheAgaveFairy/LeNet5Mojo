from std.subprocess import run  # var current = run("date")
import std.os as os
from std.reflection import reflect
from constants import act_fn


@fieldwise_init
struct LogFormat(Copyable, Movable, ImplicitlyCopyable):
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
    var sample_size: Int
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
        sample_size: Int,
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
        self.sample_size = sample_size
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
            + String(self.sample_size)
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
        self,
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
        self,
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


struct ResultLogger(LeNet5Logger, ImplicitlyCopyable):
    var output_path: String
    var format_type: LogFormat

    def __init__(
        out self, output_path: String, format_type: LogFormat = LogFormat.CSV
    ):
        self.output_path = output_path
        self.format_type = format_type.copy()

    def logInferenceResult(
        self,
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
        self,
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

    def _writeResult[T: LogEntry](self, result: T) raises -> None:
        var content = String("")
        if not os.path.exists(self.output_path):
            if self.format_type == materialize[LogFormat.CSV]():
                content += result.getHeaders() + "\n"
            else:
                content += "INVALID HEADER\n"
        if self.format_type == materialize[LogFormat.CSV]():
            content += result.toCSV() + "\n"
        else:
            content += "INVALID CONTENT\n"
        with open(self.output_path, "a") as file:
            file.write(content)


@fieldwise_init
struct MultiFileLogger(LeNet5Logger, ImplicitlyCopyable):
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
        self,
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
        self,
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
