"""A small declarative CLI parser (standalone demo; the app uses `cli`).

TODO critique (2026-07-03, bugs verified by running):
- bool flag eats the next flag: `--verbose --seed 99` -> BOTH silently vanish,
  no diagnostic (_find takes "--seed" as the value, marks it consumed). Fix:
  _find refuses values starting with "--"; better: bare bools, presence = True.
- trailing `--seed` (no value) runs the whole app on the default, THEN exits
  with a misleading late "not valid" — fail loud at parse point, not at del.
- exit() in __del__ during unwind swallowed a pending exception (the SIMD
  demo error never printed). Validate via explicit finish(); __del__ at most
  a backstop.
- `--verbose banana` silently defaults (Int/Float _fail, Bool doesn't); in
  __del__, help check should run BEFORE unknown-arg check (-h + stale flag
  currently errors instead of helping); consume by index, not string match.
- verbosity root cause: T: AnyType. Bound T: ImplicitlyCopyable & Writable
  (Bool conforms — verified) -> hoist help-append + not-found return once,
  delete both conforms_to asserts and the _register closure (~50 -> ~25
  lines); unsupported T becomes `comptime assert False` and get() drops
  `raises` (callers lose the try).
- to actually replace cli.mojo: needs bare bools + loud missing-value; range
  checks (num-streams 1..MAX) stay at the call site after get().
"""

from std.sys import argv, stderr, exit


struct ArgParser[hard_exit_mode: Bool = True]():
    """Declarative CLI parser: every `get()` reads a flag and registers its help
    line in one call. Unknown args are caught at destruction; `hard_exit_mode`
    decides whether that — and parse failures — exit the process or fall back to
    the supplied default.
    """

    var help_strs: List[String]
    var print_help: Bool
    var args: List[String]
    var consumed: List[String]

    def __init__(out self):
        self.help_strs = List[String]()
        self.print_help = False
        self.args = List[String]()
        self.consumed = type_of(self.consumed)()  # List[String] but sillier
        var raw = argv()
        for i in range(1, len(raw)):  # skip argv[0] (program name)
            var a = String(raw[i])
            if a == "-h" or a == "--help":
                self.print_help = True
            else:
                self.args.append(a)

    def get[
        T: AnyType
    ](mut self, tag: String, default: T, desc: String) raises -> T:
        comptime assert conforms_to(
            T, ImplicitlyCopyable
        ), "arg T not ImplicitlyCopyable"

        # the support of "Bool" means we can't check for Writable conformance here
        def _register(default_str: String) {read, mut self}:
            self.help_strs.append(
                String(t"{tag} [default {default_str}]; Usage: {desc}")
            )

        var found = self._find(tag)
        comptime if T == Bool:
            # comptime assert conforms_to(T, ImplicitlyCopyable), "CLIParser conformance failure"
            _register("True" if Bool(rebind_var[Bool](default)) else "False")
            if not found:
                return rebind_var[T](default)
            else:
                var v = found.value().lower()
                if v == "true":
                    return rebind_var[T](True)
                elif v == "false":
                    return rebind_var[T](False)
                else:
                    return rebind_var[T](default)

        elif T == Int:
            comptime assert conforms_to(T, Writable)
            _register(String(default))
            if not found:
                return rebind_var[T](default)
            else:
                try:
                    return rebind_var[T](Int(found.value()))
                except e:
                    self._fail(tag, String(found.value()), default)
                    return rebind_var[T](default)

        elif T == Float64:
            comptime assert conforms_to(T, Writable)
            _register(String(default))
            if not found:
                return rebind_var[T](default)
            else:
                try:
                    return rebind_var[T](Float64(found.value()))
                except e:
                    self._fail(tag, String(found.value()), default)
                    return rebind_var[T](default)

        elif T == String:
            comptime assert conforms_to(T, Writable)
            _register(String(default))
            if not found:
                return rebind_var[T](default)
            else:
                return rebind_var[T](found.value())

        else:
            raise Error(
                String(t"{reflect[T].name()} not supported by CLIParser")
            )

    def registerHelpString(mut self, str: StringSlice):
        """For anything that won't be automatically registered, use this."""
        self.help_strs.append(String(str))

    def _find(mut self, tag: String) -> Optional[String]:
        """ONLY find `tag` and return the value following it, if present."""
        for i in range(len(self.args)):
            if self.args[i] == tag and i + 1 < len(self.args):
                self.consumed.append(tag)
                self.consumed.append(self.args[i + 1])
                return self.args[i + 1]
        return None

    def _fail(self, tag: String, e: String, default: Some[Writable]):
        """Report a bad value for `tag`: exit under `hard_exit_mode`, else warn and keep `default`.
        """
        comptime if self.hard_exit_mode:
            print(tag, e, "EXITING.", file=stderr)
            exit(2)
        else:
            print(t"{tag} failed: {e}. using: {default}", file=stderr)

    def __del__(deinit self):
        """Reject any unconsumed args, then print help if `-h`/`--help` was seen.

        Mojo's ASAP destruction runs this right after the last `get()`, so
        validation needs no explicit `finalize()` call — it fires on scope exit.
        """
        for arg in self.args:
            if arg not in self.consumed:
                comptime if self.hard_exit_mode:
                    print(arg, "not valid. EXITING.", file=stderr)
                    exit(2)
                else:
                    print(arg, "not valid.", file=stderr)

        if self.print_help:
            print("How to use:")
            for hs in self.help_strs:
                print("\t", hs)
            exit(0)


def main():
    var parser = ArgParser()
    parser.registerHelpString("-D define hints could go here, etc")
    try:
        var seed = parser.get("--seed", 42, "sets seed for rand()")
        var lr = parser.get("--lr", 0.01, "learning rate")
        var name = parser.get("--name", "model", "output model name")
        var verbose = parser.get("--verbose", False, "enable verbose logging")
        print("seed:", seed)
        print("lr:", lr)
        print("name:", name)
        print("verbose:", verbose)
        var should_fail = parser.get(
            "--failure-test", SIMD[DType.float16, 4](1.0), "SIMD should fail"
        )
    except e:
        print(e)
