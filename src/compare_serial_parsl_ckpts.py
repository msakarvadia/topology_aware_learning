import torch


def compare_models(sd_1, sd_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(sd_1.items(), sd_2.items()):
        if torch.equal(key_item_1[1].cpu(), key_item_2[1].cpu()):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismtach found at", key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print("Models match perfectly! :)")


# iterate through models

for round_idx in range(5):
    print(f"{round_idx=}")
    sd_1 = torch.load(f"0_{round_idx}_before_local_train_parsl.pth", weights_only=True)
    sd_2 = torch.load(f"0_{round_idx}_before_local_train_serial.pth", weights_only=True)
    compare_models(sd_1, sd_2)
