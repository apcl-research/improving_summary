import subprocess
import re
import sys


def get_use_score(predfile, data, coms_filename):
    command = ["python3", "use_score_v.py", predfile, "--data=" + data, "--coms-filename=" + coms_filename]
    result = subprocess.run(command, capture_output=True, text=True)
    match = re.search(r'cosine similarity score with universal sentence encoder embedding is ([\d\.]+)', result.stdout)
    return float(match.group(1)) if match else None


def get_meteor_score(predfile, data, coms_filename):
    command = ["python3", "meteor.py", predfile, "--coms-filename=" + coms_filename, "--data=" + data]
    result = subprocess.run(command, capture_output=True, text=True)
    match = re.search(r'M ([\d\.]+)', result.stdout)
    return float(match.group(1)) if match else None


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 script.py <data> <coms_filename> <predfile1> <predfile2> ...")
        sys.exit(1)

    data = sys.argv[1]
    coms_filename = sys.argv[2]
    predfiles = sys.argv[3:]

    use_scores = []
    meteor_scores = []

    for predfile in predfiles:
        use_score = get_use_score(predfile, data, coms_filename)
        meteor_score = get_meteor_score(predfile, data, coms_filename)

        if use_score is not None:
            use_scores.append(use_score)
        if meteor_score is not None:
            meteor_scores.append(meteor_score)

        print(f"{predfile}: USE Score = {use_score}, METEOR Score = {meteor_score}")

    avg_use_score = sum(use_scores) / len(use_scores) if use_scores else None
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores) if meteor_scores else None

    print(f"\nAverage USE Score: {avg_use_score}")
    print(f"Average METEOR Score: {avg_meteor_score}")


if __name__ == "__main__":
    main()