import torch


class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])


def main():
    checkpoint = torch.load("llama-2-7b-chat/consolidated.00.pth", map_location="cpu")

    container = torch.jit.script(Container(checkpoint))
    container.save("llama-2-7b-chat.pt")


if __name__ == "__main__":
    main()
