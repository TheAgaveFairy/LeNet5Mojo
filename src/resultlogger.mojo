"""CSV logging of training epochs and inference runs via field reflection."""

from std.subprocess import run  # var current = run("date")
import std.os as os
from std.reflection import reflect
from constants import act_fn


# Generic CSV via reflection — header from field NAMES, row from field VALUES
# (`reflect[T].field_ref[i]` + `trait_downcast` to a Writable view; both builtins).
# EVERY field must be Writable — that's why `ftype` is materialized to a String
# below instead of kept as a DType (which reflects as lowercase "float32"). Adding
# a CSV column is now just adding a struct field; toCSV/getHeaders never change.
def reflectHeaders[T: AnyType]() -> String:
    """Comma-joined field names of `T` — the CSV header for any log-entry struct.
    """
    comptime names = reflect[T].field_names()
    var line = String("")
    comptime for i in range(reflect[T].field_count()):
        if i > 0:
            line += ","
        line += String(names[i])
    return line^


def reflectCSV[T: AnyType](ref s: T) -> String:
    """Comma-joined field values of `s` — one CSV row. Every field must be `Writable`.
    """
    var line = String("")
    comptime for i in range(reflect[T].field_count()):
        if i > 0:
            line += ","
        ref fr = reflect[T].field_ref[i](s)
        comptime assert conforms_to(type_of(fr), Writable), (
            "reflectCSV: every field of "
            + reflect[T].name()
            + " must be Writable"
        )
        line += String(trait_downcast[Writable](fr))
    return line^


@fieldwise_init
struct LogFormat(Copyable, ImplicitlyCopyable, Movable):
    """Enum-style output format tag. `CSV` is implemented; `JSON` is reserved.
    """

    var value: Int  # enum esque

    comptime CSV = LogFormat(0)
    comptime JSON = LogFormat(1)  # TODO: implement

    def __eq__(self, other: LogFormat) -> Bool:
        return self.value == other.value

    def __ne__(self, other: LogFormat) -> Bool:
        return self.value != other.value


trait LogEntry:
    """A row that can serialize itself and supply its own header."""

    def toCSV(self) -> String:
        ...

    @staticmethod
    def getHeaders() -> String:
        ...

    # TODO: add JSON


struct InferenceResult(LogEntry):
    """One inference run: device, timing, accuracy, and the config it ran under.
    """

    var timestamp: String
    var device: String
    var elapsed_ns: UInt  # perf_counter_ns() -> UInt
    var correct: Int
    var test_size: Int
    var stream_batch_size: Int
    var num_streams: Int
    var ftype: String  # materialized from DType so reflection emits "Float32"/"Float64"
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
        self.ftype = "Float64" if ftype == DType.float64 else "Float32"
        self.activation_fn = reflect[act_fn].base_name()

    def toCSV(self) -> String:
        return reflectCSV(self)

    @staticmethod
    def getHeaders() -> String:
        return reflectHeaders[Self]()


struct TrainingResult(LogEntry):
    """One training epoch: device, timing, accuracy, loss, and learning rate."""

    var timestamp: String
    var device: String
    var epoch: Int
    var elapsed_ns: UInt
    var correct: Int
    var sample_size: Int
    var loss: Float32
    var learning_rate: Float32
    var ftype: String  # materialized from DType so reflection emits "Float32"/"Float64"
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
        self.ftype = "Float64" if ftype == DType.float64 else "Float32"
        self.activation_fn = reflect[act_fn].base_name()

    def toCSV(self) -> String:
        return reflectCSV(self)

    @staticmethod
    def getHeaders() -> String:
        return reflectHeaders[Self]()


trait MyLogger:
    """Sink for training-epoch and inference-run records."""

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


comptime LeNet5Logger = MyLogger & Copyable


struct ResultLogger(ImplicitlyCopyable, LeNet5Logger):
    """Appends records to one file, writing the header first if the file is new.
    """

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
        """Append `result` as a row, prepending the header when the file is created.
        """
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
struct MultiFileLogger(ImplicitlyCopyable, LeNet5Logger):
    """Routes inference and training records to separate files under `base_path`.
    """

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
