from std.sys import argv, stderr, exit

struct ArgParser[hard_exit_mode: Bool = True]():
    var help_strs: List[String]
    var print_help: Bool
    var args: List[String]
    var consumed: List[String]

    def __init__(out self):
        self.help_strs = List[String]()
        self.print_help = False
        self.args = List[String]()
        self.consumed = type_of(self.consumed)() # List[String] but sillier
        var raw = argv()
        for i in range(1, len(raw)):  # skip argv[0] (program name)
            var a = String(raw[i])
            if a == "-h" or a == "--help":
                self.print_help = True
            self.args.append(a)

    def _find(
        mut self, tag: String, default_str: String, desc: String
    ) -> Optional[String]:
        self.help_strs.append(String(t"{tag} [default {default_str}]; Usage: {desc}"))
        for i in range(len(self.args)):
            if self.args[i] == tag and i + 1 < len(self.args):
                self.consumed.append(tag)
                self.consumed.append(self.args[i + 1])
                return self.args[i + 1]
        return None

    def _fail(self, tag: String, e: String, default: Some[Writable]):
        comptime if self.hard_exit_mode:
            print(tag, e, "EXITING.", file = stderr)
            exit(2)
        else:
            print(t"{tag} failed: {e}. using: {default}", file=stderr)

    def get(mut self, tag: String, default: Int, desc: String) -> Int:
        var found = self._find(tag, String(default), desc)
        if found:
            try:
                return Int(found.value())
            except e:
                self._fail(tag, String(e), default)
        return default

    def get(mut self, tag: String, default: Float64, desc: String) -> Float64:
        var found = self._find(tag, String(default), desc)
        if found:
            try:
                return Float64(found.value())
            except e:
                self._fail(tag, String(e), default)
        return default

    def get(mut self, tag: String, default: String, desc: String) -> String:
        var found = self._find(tag, default, desc)
        if found:
            return found.value()
        return default

    def get(mut self, tag: String, default: Bool, desc: String) -> Bool:
        var found = self._find(tag, String(default), desc)
        if found:
            var v = found.value().lower()
            if v == "true" or v == "1" or v == "yes":
                return True
            if v == "false" or v == "0" or v == "no":
                return False
            self._fail(tag, "'v' not boolable", default)
        return default

    def __del__(deinit self):
        for arg in self.args:
            if arg not in self.consumed:
                comptime if self.hard_exit_mode:
                    print(arg, "not valid. EXITING.", file = stderr)
                    exit(2)
                else:
                    print(arg, "not valid.", file = stderr)

        if self.print_help:
            print("How to use:")
            for hs in self.help_strs:
                print("\t", hs)
            exit(0)


def main() raises:
    var parser = ArgParser()
    var seed = parser.get("--seed", 42, "sets seed for rand()")
    var lr = parser.get("--lr", 0.01, "learning rate")
    var name = parser.get("--name", "model", "output model name")
    var verbose = parser.get("--verbose", False, "enable verbose logging")
    print("seed:", seed)
    print("lr:", lr)
    print("name:", name)
    print("verbose:", verbose)
