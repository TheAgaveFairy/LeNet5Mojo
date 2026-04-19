def showProgress(progress: Int, total: Int) -> None:
    comptime bar_width = 50
    var ratio = Float64(progress) / Float64(total)
    var filled = Int(Float64(bar_width) * ratio)
    # print(chr(27) + "[2J",end="")
    print("\r[", end="")
    for _ in range(filled):
        print("=", end="")
    for _ in range(filled, bar_width):
        print(" ", end="")
    print("]", round(ratio * 100, 3), "%", end="")
