import torch

class Truncation:
    def __init__(
        self,
        content: int = 151655,
        pad: int = 151643,
        start: int = 151652,
        end: int = 151653,
        train: bool = False,
    ):
        self.content = content
        self.pad = pad
        self.start = start
        self.end = end
        self.train = train

    def count_start(
        self, 
        input_ids: torch.Tensor, 
    ) -> torch.Tensor:
        return torch.sum(input_ids == self.start, dim=1)

    def replace_last_image(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        # remove the last image token and its content if it's truncated
        cumstart = torch.cumsum((input_ids == self.start).to(torch.int), dim=1)
        cumend = torch.cumsum((input_ids == self.end).to(torch.int), dim=1)
        mask_last_image = cumstart == cumstart[:, -1].unsqueeze(1)
        has_unfinished_img = cumstart[:, -1].unsqueeze(1) != cumend[:, -1].unsqueeze(1)
        input_ids[has_unfinished_img & mask_last_image] = self.pad
        attention_mask[has_unfinished_img & mask_last_image] = 0
        return input_ids, attention_mask

    def sanity_check(
        self, 
        input_ids, 
        attention_mask, 
        pixel_values, 
        image_grid_thw
    ):
        # sanity check of the inputs
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask cannot be None")
        if pixel_values is None and image_grid_thw is None:
            if (
                torch.sum(input_ids == self.start).item()
                or torch.sum(input_ids == self.end).item()
                or torch.sum(input_ids == self.content).item()
            ):
                raise ValueError(
                    "The input_ids contains <|vision_start|>, <|vision_end|> or <|image_pad|> tokens but no pixel_values or image_grid_thw is provided"
                )
            return
        if (pixel_values is None and image_grid_thw is not None) or (
            pixel_values is not None and image_grid_thw is None
        ):
            raise ValueError(
                "Both pixel_values and image_grid_thw must be provided or not provided at the same time"
            )
        if (
            torch.sum(input_ids == self.start).item()
            != torch.sum(input_ids == self.end).item()
        ):
            raise ValueError(
                "The number of <|vision_start|> does not match the number of <|vision_end|>"
            )
        if torch.sum(input_ids == self.start).item() != image_grid_thw.shape[0]:
            raise ValueError(
                "The number of <|vision_start|> tokens does not match the number of images in the image_grid_thw"
            )
        if torch.sum(input_ids == self.content).item() * 4 != pixel_values.shape[0]:
            raise ValueError(
                "The number of pixel_values does not match the number of <|image_pad|> tokens in the input_ids"
            )
        if pixel_values.shape[0] != torch.sum(torch.prod(image_grid_thw, dim=1)).item():
            raise ValueError(
                "The number of pixel_values does not match the number of images in the image_grid_thw"
            )

    def add_black_image(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
    ):
        # there is no padding token in the first example
        if torch.argwhere(input_ids[0] == self.pad).shape[0] == 0:
            input_ids[0, -6:] = self.content
            input_ids[0, -6] = self.start
            input_ids[0, -1] = self.end
            attention_mask[0, -6:] = 0
        # there is padding token in the first example, but it's after the last sixth token
        elif (
            torch.argwhere(input_ids[0] == self.pad)[0][0].item() + 6
            > input_ids.shape[-1]
        ):
            input_ids[0, -6:] = self.content
            input_ids[0, -6] = self.start
            input_ids[0, -1] = self.end
            attention_mask[0, -6:] = 0
        else:
            first_padding_pos = torch.argwhere(input_ids[0] == self.pad)[0][0].item()
            input_ids[0, first_padding_pos : first_padding_pos + 6] = self.content
            input_ids[0, first_padding_pos] = self.start
            input_ids[0, first_padding_pos + 5] = self.end
            attention_mask[0, first_padding_pos : first_padding_pos + 6] = 0
        # replace the last 6 rokens black of the first example of the input_ids with image, add a black image

        pixel_values = (
            torch.cat(
                [
                    torch.ones((16, 392)) * -1.7923,
                    torch.ones((16, 392)) * -1.7521,
                    torch.ones((16, 392)) * -1.4802,
                ],
                dim=-1,
            )
            .to(torch.bfloat16)
            .to(input_ids.device)
        )
        image_grid_thw = torch.tensor([[1, 4, 4]]).to(input_ids.device)
        return input_ids, attention_mask, pixel_values, image_grid_thw

    def truncate(
        self, 
        inputs: dict, 
        length: int = 32768
    ) -> dict:
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        # sanity check before truncation
        self.sanity_check(input_ids, attention_mask, pixel_values, image_grid_thw)

        # calculate raw_num_images
        raw_num_images = self.count_start(input_ids)

        # truncate the input_ids and attention_mask
        input_ids = input_ids[:, :length]
        attention_mask = attention_mask[:, :length]

        # if the input_ids is not truncated, return the original inputs or add an additional image
        if pixel_values is None and image_grid_thw is None:
            if self.train:
                input_ids, attention_mask, pixel_values, image_grid_thw = (
                    self.add_black_image(input_ids, attention_mask)
                )
                self.sanity_check(
                    input_ids, attention_mask, pixel_values, image_grid_thw
                )
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                }
            else:
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": None,
                    "image_grid_thw": None,
                }

        # replace the last truncated images with padding
        input_ids, attention_mask = self.replace_last_image(input_ids, attention_mask)

        # calculate new_num_images
        new_num_images = self.count_start(input_ids)

        # truncate the pixel_values and image_grid_thw
        if input_ids.shape[0] == 1:
            cum_num_images = torch.tensor([0]).to(input_ids.device)
        else:
            cum_num_images = torch.cat(
                [
                    torch.tensor([0]).to(input_ids.device),
                    torch.cumsum(raw_num_images, dim=0)[:-1],
                ]
            )

        if image_grid_thw.shape[0] == 1:
            cum_num_pixels = torch.tensor([0]).to(input_ids.device)
        else:
            cum_num_pixels = torch.cat(
                [
                    torch.tensor([0]).to(input_ids.device),
                    torch.cumsum(
                        image_grid_thw[:, 0]
                        * image_grid_thw[:, 1]
                        * image_grid_thw[:, 2],
                        dim=0,
                    )[:-1],
                ]
            )

        selected_images_ids = torch.cat(
            [
                torch.arange(cum_num_images[i], cum_num_images[i] + new_num_images[i])
                for i in range(input_ids.shape[0])
            ]
        )
        if selected_images_ids.shape[0] == 0:
            if self.train:
                input_ids, attention_mask, pixel_values, image_grid_thw = self.add_black_image(
                    input_ids, attention_mask
                )
                self.sanity_check(
                    input_ids, attention_mask, pixel_values, image_grid_thw
                )
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                }
            else:
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": None,
                    "image_grid_thw": None,
                }

        selected_pixel_ids = torch.cat(
            [
                torch.arange(
                    cum_num_pixels[i],
                    cum_num_pixels[i]
                    + image_grid_thw[i, 0]
                    * image_grid_thw[i, 1]
                    * image_grid_thw[i, 2],
                )
                for i in selected_images_ids
            ]
        )

        image_grid_thw = image_grid_thw[selected_images_ids]
        pixel_values = pixel_values[selected_pixel_ids]

        assert torch.sum(new_num_images).item() == image_grid_thw.shape[0]
        assert input_ids.shape == attention_mask.shape
        self.sanity_check(input_ids, attention_mask, pixel_values, image_grid_thw)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }