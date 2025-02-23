def get_epoch(yaml_path: str) -> int:
    epoch = 1
    try:
        with open(yaml_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip().startswith("epochs:"):
                    try:
                        epoch = int(line.split(":")[1].strip())
                    except ValueError:
                        epoch = 1
                    break

    except FileNotFoundError:
        print("File not found")

    return epoch - 1  # return the last epoch. Note that the epoch starts from 0
