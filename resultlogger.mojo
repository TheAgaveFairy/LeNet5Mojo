from std.subprocess import run # var current = run("date")
import std.os as os

@fieldwise_init
struct LogFormat(Copyable, Movable):
    var value: Int # enum esque

    comptime CSV = LogFormat(0)
    comptime JSON = LogFormat(1) # TODO: implement

    def __eq__(self, other: LogFormat) -> Bool:
        return self.value == other.value

    def __ne__(self, other: LogFormat) -> Bool:
        return self.value != other.value

trait LogEntry:
    def toCSV(self) -> String: ...
    @staticmethod
    def getHeaders() -> String: ...
    # TODO: add JSON

struct InferenceResult(LogEntry):
    var timestamp: String
    var device: String
    var elapsed_ns: UInt
    var correct: UInt
    var test_size: UInt
    var batch_size: UInt
    var ftype: DType
    
    def __init__(out self, device: String, elapsed_ns: UInt, correct: UInt, test_size: UInt, batch_size: UInt, ftype: DType):
        try:
            self.timestamp = run("date") # subprocess
        except e:
            print("InferenceResult timestamp error:", e)
            self.timestamp = "TIMESTAMP FAILED"

        self.device = device
        self.elapsed_ns = elapsed_ns
        self.correct = correct
        self.test_size = test_size
        self.batch_size = batch_size
        self.ftype = ftype

    def toCSV(self) -> String:
        var datatype_string = "Float32"
        if self.ftype == DType.float64:
            datatype_string = "Float64"
        return (self.timestamp + "," + 
            self.device + "," + 
            String(self.elapsed_ns) + "," + 
            String(self.correct) + "," + 
            String(self.test_size) + "," + 
            String(self.batch_size) + "," +
            datatype_string)

    @staticmethod
    def getHeaders() -> String:
        return "timestamp,device,elapsed_ns,correct,test_size,batch_size,ftype"
        
struct TrainingResult(LogEntry):
    var timestamp: String
    var device: String
    var epoch: UInt
    var elapsed_ns: UInt
    var correct: UInt
    var test_size: UInt # kinda just gonna be the batch_size since CPU only for now # TODO:
    var loss: Float32
    var learning_rate: Float32
    var ftype: DType

    def __init__(out self, device: String, epoch: UInt, elapsed_ns: UInt, correct: UInt, test_size: UInt, loss: Float32, learning_rate: Float32, ftype: DType):
        try:
            self.timestamp = run("date") # subprocess
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

    def toCSV(self) -> String:
        var datatype_string = "Float32"
        if self.ftype == DType.float64:
            datatype_string = "Float64"
        return (self.timestamp + "," + 
            self.device + "," + 
            String(self.epoch) + "," + 
            String(self.elapsed_ns) + "," + 
            String(self.correct) + "," + 
            String(self.test_size) + "," + 
            String(self.loss) + "," + 
            String(self.learning_rate) + "," + 
            datatype_string)
    
    @staticmethod
    def getHeaders() -> String:
        return "timestamp,device,epoch,elapsed_ns,correct,test_size,loss,learning_rate,ftype"

trait MyLogger:
    def logInferenceResult(mut self, device: String, elapsed_ns: UInt, correct: UInt, test_size: UInt, batch_size: UInt, ftype: DType) raises -> None: ...
    def logTrainingEpoch(mut self, device: String, epoch: UInt, elapsed_ns: UInt, correct: UInt, test_size: UInt, loss: Float32, learning_rate: Float32, ftype: DType) raises -> None: ...

comptime LeNet5Logger = MyLogger & Copyable & Movable

@fieldwise_init
struct ResultLogger(LeNet5Logger):
    var output_path: String
    var format_type: LogFormat
    var headers_written: Bool # TODO: make this better

    def __init__(out self, output_path: String, format_type: LogFormat = LogFormat.CSV):
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

    def logInferenceResult(mut self, device: String, elapsed_ns: UInt, correct: UInt, test_size: UInt, batch_size: UInt, ftype: DType) raises -> None:
        var result = InferenceResult(device, elapsed_ns, correct, test_size, batch_size, ftype)
        self._writeResult(result)

    def logTrainingEpoch(mut self, device: String, epoch: UInt, elapsed_ns: UInt, correct: UInt, test_size: UInt, loss: Float32, learning_rate: Float32, ftype: DType) raises -> None:
        var result = TrainingResult(device, epoch, elapsed_ns, correct, test_size, loss, learning_rate, ftype)
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

        #self._appendToFile(content)
        with open(self.output_path, "a") as file:
            file.write(content)
    
    @deprecated("can open files in append mode now")
    def _appendToFile(mut self, content: String) raises -> None:
        var existing: String

        if not os.path.exists(self.output_path):
            with open(self.output_path, "w") as f:
                pass
        with open(self.output_path, "r") as file:
            existing = file.read()
        with open(self.output_path, "w") as file:
            file.write(existing + content)

@fieldwise_init
struct MultiFileLogger(LeNet5Logger):
    var base_path: String
    var format_type: LogFormat
    var inference_logger: ResultLogger
    var training_logger: ResultLogger
    
    def __init__(out self, base_path: String, format: LogFormat = LogFormat.CSV):
        self.base_path = base_path
        self.format_type = format.copy()
        
        var ext = ".csv" if format == materialize[LogFormat.CSV]() else (".json" if format == materialize[LogFormat.JSON]() else ".tsv")
        
        self.inference_logger = ResultLogger(base_path + "inference" + ext, format)
        self.training_logger = ResultLogger(base_path + "training" + ext, format)
    
    def logInferenceResult(mut self, device: String, elapsed_ns: UInt, correct: UInt, test_size: UInt, batch_size: UInt, ftype: DType) raises -> None:
        self.inference_logger.logInferenceResult(device, elapsed_ns, correct, test_size, batch_size, ftype)
    
    def logTrainingEpoch(mut self, device: String, epoch: UInt, elapsed_ns: UInt, correct: UInt, test_size: UInt, loss: Float32, learning_rate: Float32, ftype: DType) raises -> None:
        self.training_logger.logTrainingEpoch(device, epoch, elapsed_ns, correct, test_size, loss, learning_rate, ftype)

def main() raises:
    comptime output_path = "results/"
    var logger = MultiFileLogger(output_path, materialize[LogFormat.CSV]())
    
    logger.logInferenceResult("RTX6069", 420, 99, 100, 10, DType.float64)
    logger.logInferenceResult("GTX730", 9001, 98, 100, 10, DType.float32)

    logger.logTrainingEpoch("7600X", 1, 69, 11, 100, 0.1337, 0.10, DType.float32)
    print("Results logged at", output_path)
