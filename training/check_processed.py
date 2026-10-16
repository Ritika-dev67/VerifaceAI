import os

for split in ["train", "val", "test"]:
    print("\n", split.upper())

    for category in ["real", "fake"]:

        folder = f"../processed/{split}/{category}"

        count = len(os.listdir(folder))

        print(category, ":", count)