import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))

EDITS = {
    "ppo_train.py": [
        ('VARIANTS = ("baseline", "risk_only", "uncertainty_only", "full")',
         'VARIANTS = ("baseline", "risk_only", "uncertainty_only", "full", "mc_nocaution")'),
        ('    k_train = n_mc if use_unc else 1',
         '    use_mc = use_unc or variant == "mc_nocaution"   # 15-pass average, with or without caution\n'
         '    k_train = n_mc if use_mc else 1'),
    ],
    "run_seeds.py": [
        ('VARIANTS = ["baseline", "risk_only", "uncertainty_only", "full"]',
         'VARIANTS = ["mc_nocaution"]'),
        ('SEEDS = [1, 2, 3, 4]',
         'SEEDS = [0, 1, 2, 3, 4]'),
    ],
}


def patch(name, edits):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        print(f"{name}: NOT FOUND in {HERE}. Nothing changed.")
        return False
    raw = open(path, "rb").read()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8").replace("\r\n", "\n")

    if all(new in text for _, new in edits):
        print(f"{name}: already patched, nothing to do.")
        return True

    for old, _ in edits:
        n = text.count(old)
        if n != 1:
            print(f"{name}: expected exactly one occurrence of this line but found {n}:\n    {old}\n"
                  f"Nothing was changed in {name}. Send me this message.")
            return False

    for old, new in edits:
        text = text.replace(old, new)

    if not os.path.exists(path + ".bak"):
        shutil.copyfile(path, path + ".bak")
    if crlf:
        text = text.replace("\n", "\r\n")
    with open(path, "wb") as f:
        f.write(text.encode("utf-8"))
    print(f"{name}: patched (backup saved as {name}.bak)")
    return True


def main():
    ok = patch("ppo_train.py", EDITS["ppo_train.py"])
    if ok:
        patch("run_seeds.py", EDITS["run_seeds.py"])
    else:
        print("run_seeds.py was not touched because ppo_train.py could not be patched.")


if __name__ == "__main__":
    main()