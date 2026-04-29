"""Edit an existing image with a text caption.

Same `image-generation` model as `image_generation.py`, but the user
message carries both a text caption and an `ImagePart` so duck.ai
performs an edit instead of a fresh generation.
"""

from duck_ai import DuckChat, image_generation


def main() -> None:
    with DuckChat(model=image_generation) as duck:
        data = duck.edit_image(
            "make the duck wear a tiny chef hat and hold a wooden spoon",
            "duck.jpg",
            save_to="duck_chef.jpg",
        )
        print(f"saved {len(data)} bytes -> duck_chef.jpg")


if __name__ == "__main__":
    main()
