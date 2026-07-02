from std.algorithm.functional import vectorize
from layout import Layout, LayoutTensor
from std.math import tanh, exp, sqrt, erf, log, pi, tau

from constants import ftype, sftype, nelts


# comptime activation_fn = fn(sftype) -> sftype # if was scalar
trait ActivationFunction:
    """The compile-time-swappable activation interface. A conformer supplies the
    SIMD `simdForward`/`simdBackward` kernels; `forward`/`backward` map them over a
    tensor and default to a plain elementwise pass.
    """

    @staticmethod
    @always_inline("nodebug")
    def forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        """Operates in-place. Default: map `simdForward` over the tensor.

        Every activation's forward is just its `simdForward` applied elementwise,
        so this default covers all of them — a struct only overrides it if it needs
        something other than a pure elementwise map.
        """

        def vectorize_closure[width: Int](i: Int) {read}:
            var nums = x.ptr.load[width=width](i)
            x.ptr.store[width=width](i, Self.simdForward(nums))

        vectorize[nelts](comptime (layout.size()), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    def backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_output: LayoutTensor[ftype, layout, _],
        d_z: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """Takes the original forward input 'x',
        the upstream gradient, d_output,
        and calculates the d_z gradient as our output for pre-act.
        """
        ...

    @staticmethod
    @always_inline("nodebug")
    def simdForward[
        fp: DType, width: Int
    ](x: SIMD[fp, width]) -> SIMD[fp, width]:
        """SIMD forward pass parameterized over dtype and vector width."""
        ...

    @staticmethod
    @always_inline("nodebug")
    def simdBackward[
        fp: DType, width: Int
    ](x: SIMD[fp, width], d_output: SIMD[fp, width]) -> SIMD[fp, width]:
        """SIMD backward pass parameterized over dtype and vector width.
        x is the pre-activation input, d_output is the upstream gradient.
        """
        ...


struct ReLU(ActivationFunction):
    """
    ReLU(x) = max(0, x)
    """

    @staticmethod
    @always_inline("nodebug")
    def backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_output: LayoutTensor[ftype, layout, _],
        d_z: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """
        SCALAR FORM is "return d_output if x > 0.0 else 0.0".
        SIMD enhanced.
        """

        def closure[width: Int](i: Int) {read}:
            comptime zeros = SIMD[ftype, width](0.0)
            var xvec = x.ptr.load[width](i)
            var mask = xvec.gt(zeros)
            var vec = d_output.ptr.load[width](i)
            var res = mask.select(vec, zeros)
            d_z.ptr.store[width](i, res)

        vectorize[nelts](comptime (layout.size()), closure)

    @staticmethod
    @always_inline("nodebug")
    def simdForward[
        fp: DType, width: Int
    ](x: SIMD[fp, width]) -> SIMD[fp, width]:
        comptime assert (
            fp.is_floating_point()
        ), "simdForward requires floating point"
        comptime zeros = SIMD[fp, width](0.0)
        return x.gt(zeros).select(x, zeros)

    @staticmethod
    @always_inline("nodebug")
    def simdBackward[
        fp: DType, width: Int
    ](x: SIMD[fp, width], d_output: SIMD[fp, width]) -> SIMD[fp, width]:
        comptime assert (
            fp.is_floating_point()
        ), "simdBackward requires floating point"
        comptime zeros = SIMD[fp, width](0.0)
        return x.gt(zeros).select(d_output, zeros)


struct Sigmoid(ActivationFunction):
    """
    Exact implementation.
    sigmoid(x) = 1 / (1 + e^(-x))
    """

    @staticmethod
    @always_inline("nodebug")
    def backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_output: LayoutTensor[ftype, layout, _],
        d_z: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """
        sigmoid'(z) = sigmoid(z) * (1-sigmoid(z))
        """

        def vectorize_closure[width: Int](i: Int) {read}:
            var nums = x.ptr.load[width=width](i)
            comptime ones = SIMD[ftype, width](1.0)
            var upstream = d_output.ptr.load[width=width](i)
            var s = ones / (ones + exp(-nums))
            var answer = upstream * s * (ones - s)
            d_z.ptr.store[width=width](i, answer)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    def simdForward[
        fp: DType, width: Int
    ](x: SIMD[fp, width]) -> SIMD[fp, width]:
        comptime assert (
            fp.is_floating_point()
        ), "simdForward requires floating point"
        comptime ones = SIMD[fp, width](1.0)
        return ones / (ones + exp(-x))

    @staticmethod
    @always_inline("nodebug")
    def simdBackward[
        fp: DType, width: Int
    ](x: SIMD[fp, width], d_output: SIMD[fp, width]) -> SIMD[fp, width]:
        comptime assert (
            fp.is_floating_point()
        ), "simdBackward requires floating point"
        comptime ones = SIMD[fp, width](1.0)
        var s = ones / (ones + exp(-x))
        return d_output * s * (ones - s)


struct Tanh(ActivationFunction):
    """
    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    """

    @staticmethod
    @always_inline("nodebug")
    def backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_output: LayoutTensor[ftype, layout, _],
        d_z: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """
        tanh'(x) = 1 - tanh(x)^2
        """

        def vectorize_closure[width: Int](i: Int) {read}:
            comptime ones = SIMD[ftype, width](1.0)
            var t = tanh(x.ptr.load[width=width](i))
            var upstream = d_output.ptr.load[width=width](i)
            d_z.ptr.store[width=width](i, upstream * (ones - t * t))

        vectorize[nelts](comptime (layout.size()), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    def simdForward[
        fp: DType, width: Int
    ](x: SIMD[fp, width]) -> SIMD[fp, width]:
        comptime assert (
            fp.is_floating_point()
        ), "simdForward requires floating point"
        return tanh(x)

    @staticmethod
    @always_inline("nodebug")
    def simdBackward[
        fp: DType, width: Int
    ](x: SIMD[fp, width], d_output: SIMD[fp, width]) -> SIMD[fp, width]:
        comptime assert (
            fp.is_floating_point()
        ), "simdBackward requires floating point"
        comptime ones = SIMD[fp, width](1.0)
        var t = tanh(x)
        return d_output * (ones - t * t)


struct GELU(ActivationFunction):
    """
    Exact implementation.
    GELU(x) = x * CDF(x) = x * (1 + erf(x / sqrt(2))) / 2
    Can be approximated with a tanh version, or quick version (see GELU paper).
    """

    @staticmethod
    @always_inline("nodebug")
    def backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_output: LayoutTensor[ftype, layout, _],
        d_z: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """
        (x * CDF(x))' = x'CDF(x) + xCDF'(x) .
                      = CDF(x) + xPDF(x)    .

        PyTorch:
        pdf_val = torch.distributions.Normal(0, 1).log_prob(data).exp() .
        return grad_output * (cdf + data * pdf_val)                     .

        Modifies d_input in-place, assuming it was already loaded with d_output.
        This approach is less explicit and more error prone, but a touch faster.
        """
        comptime sqrt2 = sqrt(2.0)
        comptime sqrttau = sqrt(tau)  # math.pi * 2.0

        def vectorize_closure[width: Int](i: Int) {read}:
            var nums = x.ptr.load[width=width](i)
            comptime sqrt2_vec = SIMD[ftype, width](sqrt2)
            comptime sqrttau_vec = SIMD[ftype, width](sqrttau)
            comptime term = log(sqrttau_vec)
            comptime halves = SIMD[ftype, width](0.5)
            comptime neg_halves = SIMD[ftype, width](-0.5)
            comptime ones = SIMD[ftype, width](1.0)
            comptime inverse_sqrttau = SIMD[ftype, width](1 / sqrttau)
            var cdf = halves * (ones + erf(nums / sqrt2_vec))

            # var pdf = exp(
            #    neg_halves * nums * nums - term
            # )  # or exp(-0.5 * x**2) / sqrt(tau)
            var pdf = exp(neg_halves * nums * nums) * inverse_sqrttau

            var upstream = d_output.ptr.load[width=width](i)
            var answer = upstream * (cdf + nums * pdf)
            d_z.ptr.store[width=width](i, answer)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    def simdForward[
        fp: DType, width: Int
    ](x: SIMD[fp, width]) -> SIMD[fp, width]:
        comptime assert (
            fp.is_floating_point()
        ), "simdForward requires floating point"
        comptime typ = SIMD[fp, width]
        comptime sqrt2 = typ(sqrt(2.0))
        comptime half = typ(0.5)
        comptime one = typ(1.0)
        return half * x * (one + erf(x / sqrt2))

    @staticmethod
    @always_inline("nodebug")
    def simdBackward[
        fp: DType, width: Int
    ](x: SIMD[fp, width], d_output: SIMD[fp, width]) -> SIMD[fp, width]:
        comptime assert (
            fp.is_floating_point()
        ), "simdBackward requires floating point"
        comptime typ = SIMD[fp, width]
        comptime sqrt2 = typ(sqrt(2.0))
        comptime half = typ(0.5)
        comptime one = typ(1.0)
        comptime inv_sqrttau = typ(1.0 / sqrt(tau))
        var cdf = half * (one + erf(x / sqrt2))
        var pdf = exp(-half * x * x) * inv_sqrttau
        return d_output * (cdf + x * pdf)


struct GELUTanh(ActivationFunction):
    """
    Tanh approximation.
    GELUTanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    .
    """

    @staticmethod
    @always_inline("nodebug")
    def backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_output: LayoutTensor[ftype, layout, _],
        d_z: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """
        Defined as the following...
        k = sqrt(2 / pi)
        c = 0.044715
        t = tanh(z) = tanh(k * (x + c * x^3))
        dy/dx = 0.5 * ((1 + t) + x * (1 - t^2) * k * (1 + 3 * c * x^2))
        Doc ends with a period.
        """
        comptime k = sqrt(2.0 / pi)
        comptime c = 0.044715

        def vectorize_closure[width: Int](i: Int) {read}:
            var nums = x.ptr.load[width=width](i)
            comptime ks = SIMD[ftype, width](k)
            comptime cs = SIMD[ftype, width](c)
            comptime threecs = SIMD[ftype, width](3.0 * c)
            comptime ones = SIMD[ftype, width](1.0)
            comptime halves = SIMD[ftype, width](0.5)

            var ts = tanh(ks * (nums + cs * (nums * nums * nums)))
            var deriv = halves * (
                (ones + ts)
                + nums
                * (ones - (ts * ts))
                * ks
                * (ones + threecs * nums * nums)
            )
            var upstream = d_output.ptr.load[width=width](i)
            var answer = upstream * deriv
            d_z.ptr.store[width=width](i, answer)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    def simdForward[
        fp: DType, width: Int
    ](x: SIMD[fp, width]) -> SIMD[fp, width]:
        comptime assert (
            fp.is_floating_point()
        ), "simdForward requires floating point"
        comptime k = SIMD[fp, width](sqrt(2.0 / pi))
        comptime c = SIMD[fp, width](0.044715)
        comptime half = SIMD[fp, width](0.5)
        comptime ones = SIMD[fp, width](1.0)
        return half * x * (ones + tanh(k * (x + c * x * x * x)))

    @staticmethod
    @always_inline("nodebug")
    def simdBackward[
        fp: DType, width: Int
    ](x: SIMD[fp, width], d_output: SIMD[fp, width]) -> SIMD[fp, width]:
        comptime assert (
            fp.is_floating_point()
        ), "simdBackward requires floating point"
        comptime k = SIMD[fp, width](sqrt(2.0 / pi))
        comptime c = SIMD[fp, width](0.044715)
        comptime three_c = SIMD[fp, width](3.0 * 0.044715)
        comptime half = SIMD[fp, width](0.5)
        comptime ones = SIMD[fp, width](1.0)
        var t = tanh(k * (x + c * x * x * x))
        var deriv = half * (
            (ones + t) + x * (ones - t * t) * k * (ones + three_c * x * x)
        )
        return d_output * deriv


struct GELUFast(ActivationFunction):
    """
    FAST implementation. NOT exact - use GELU.
    GELUFast(x) = x * sigmoid(1.702 * x) https://arxiv.org/pdf/1606.08415
    .
    """

    @staticmethod
    @always_inline("nodebug")
    def _sigmoid[
        stype: DType, width: Int
    ](x: SIMD[stype, width]) -> SIMD[stype, width]:
        """SIMD Sigmoid that accepts any floating point type, not just ftype."""
        comptime assert (
            stype.is_floating_point()
        ), "_sigmoid requires floating points"
        comptime ones = SIMD[stype, width](1.0)
        comptime neg_ones = SIMD[stype, width](-1.0)
        var input = x * neg_ones
        return ones / (ones + exp(input))

    @staticmethod
    @always_inline("nodebug")
    def backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_output: LayoutTensor[ftype, layout, _],
        d_z: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """
        Defined as:
        f'(1.702x) = sigmoid(1.702x) + (x * 1.702 * sigmoid(1.702x) * (1 - sigmoid(1.702x)))
        See paper.
        """
        comptime alpha = sftype(1.702)

        def vectorize_closure[width: Int](i: Int) {read}:
            comptime alphas = SIMD[ftype, width](alpha)
            comptime ones = SIMD[ftype, width](1.0)
            var nums = x.ptr.load[width=width](i)
            var s = Self._sigmoid(nums * alphas)
            var deriv = s + alphas * nums * s * (ones - s)
            var upstream = d_output.ptr.load[width=width](i)
            var answer = upstream * deriv
            d_z.ptr.store[width=width](i, answer)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    def simdForward[
        fp: DType, width: Int
    ](x: SIMD[fp, width]) -> SIMD[fp, width]:
        comptime assert (
            fp.is_floating_point()
        ), "simdForward requires floating point"
        comptime alpha = SIMD[fp, width](1.702)
        return x * Self._sigmoid(alpha * x)

    @staticmethod
    @always_inline("nodebug")
    def simdBackward[
        fp: DType, width: Int
    ](x: SIMD[fp, width], d_output: SIMD[fp, width]) -> SIMD[fp, width]:
        comptime assert (
            fp.is_floating_point()
        ), "simdBackward requires floating point"
        comptime alpha = SIMD[fp, width](1.702)
        comptime ones = SIMD[fp, width](1.0)
        var s = Self._sigmoid(alpha * x)
        return d_output * (s + alpha * x * s * (ones - s))
