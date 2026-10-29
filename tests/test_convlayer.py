import torch
import math
import matplotlib.pyplot as plt
from filter_kernels.conv_layer import ConvLayer

def test_mult(x):
    a = ConvLayer(
        weight=torch.randn((3, 2, 7)),
        bias=torch.randn((3,))
    )
    b = ConvLayer(
        weight=torch.randn((4, 3, 7)),
        bias=torch.randn((4,))
    )

    return (a * b)(x) - b(a(x))

def test_add(x):
    a = ConvLayer(
        weight=torch.randn((3, 2, 7)),
        bias=torch.randn((3,))
    )
    b = ConvLayer(
        weight=torch.randn((3, 2, 7)),
        bias=torch.randn((3,))
    )
    return (a + b)(x) - (a(x) + b(x))

def test_l_distrib(x):
    a = ConvLayer(
        weight=torch.randn((3, 2, 7)),
        bias=torch.randn((3,))
    )
    b = ConvLayer(
            weight=torch.randn((4, 3, 7)),
            bias=torch.randn((4,))
        )
    c = ConvLayer(
        weight=torch.randn((4, 3, 7)),
        bias=torch.randn((4,))
    )

    f1 = a * (b + c)
    f2 = a * b + a * c
    return f1(x) - f2(x)

def test_r_distrib(x):
    a = ConvLayer(
        weight=torch.randn((3, 2, 7)),
        bias=torch.randn((3,))
    )
    b = ConvLayer(
            weight=torch.randn((3, 2, 7)),
            bias=torch.randn((3,))
        )
    c = ConvLayer(
        weight=torch.randn((4, 3, 7)),
        bias=torch.randn((4,))
    )

    f1 = (a + b) * c
    f2 = a * c + b * c
    return f1(x) - f2(x)

if __name__ == "__main__":
    TESTS = [
        test_mult,
        test_add,
        test_l_distrib, 
        test_r_distrib
    ]
    for test in TESTS:
        x = torch.randn((2, 40))
        error = test(x)
        fig, ax = plt.subplots()
        ax.imshow(error)
        ax.set_title(f"Error for {test.__name__}")

        def _fmt(x):
            return round(x.item(), 2)
        
        plt.text(x=0, y=15, s=f"""
            median={_fmt(error.median())}
            range=[{_fmt(error.max())}, {_fmt(error.min())}]
        """)
        plt.show()