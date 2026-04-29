from topicgpt_python.utils import *
import argparse
import re


def parse_first_topic(response):
    response = str(response)
    clean = response.replace("**","")
    clean = clean.replace("*","")
    m = re.search(r"\[(\d+)\]\s*([^:\n\r\(\*]+?):", clean)
    # Case 1: [N] Name: desc  (standard format, name and colon on same line)
    if m:
        name = m.group(2).strip()
        if name and not re.match(
            r"(?i)^(assignment reasoning|topic label|supporting quote|note|here|the|this)",
            name,
        ):
            return name
    # Case 2: [N] Name\n  (name alone on its line, colon missing or on next line)
    m2 = re.search(r"\[(\d+)\]\s*([^\n\r\[]+)\n", clean)
    if m2:
        name = m2.group(2).strip().rstrip(":").strip()
        if name:
            return name

    return " "

def metric_calc(data_file, ground_truth_col, output_col):
    """
    Calculate alignment metrics between predicted topics and ground-truth topics.

    Parameters:
    - data_file (str): Path to data file (containing both ground-truth and predicted topics)
    - ground_truth_col (str): Column name for ground-truth topics
    - output_col (str): Column name for predicted topics
    """
    # Load data
    data = pd.read_json(data_file, lines=True)
    output_topics = data[output_col]
    print(len(output_topics))

    output_topics = [parse_first_topic(topic) for topic in output_topics]
    for i in range(len(output_topics)):
        if output_topics[i] == " ":
            print("indice: ",i)
    n_empty = sum(1 for t in output_topics if t == " ")
    if n_empty > 0:
        print(f"Warning: {n_empty}/{len(output_topics)} responses could not be parsed.")

    print(len(output_topics))
    data["parsed_output"] = output_topics

    # Only retain the first topic in the list of topics
    """ output_pattern = r"\[(?:\d+)\] ([^:]+): (?:.+)"
    output_topics = [re.findall(output_pattern, topic) for topic in output_topics]
    output_topics = [topic[0] if (topic != []) else " " for topic in output_topics]
    print(len(output_topics))
    data["parsed_output"] = output_topics """

    harmonic_purity, ari, mis = calculate_metrics(
        ground_truth_col, "parsed_output", data
    )

    print("--------------------")
    print("Alignment between predicted topics and ground truth:")
    print("Harmonic Purity: ", harmonic_purity)
    print("ARI: ", ari)
    print("MIS: ", mis)
    print("--------------------")

    return calculate_metrics(ground_truth_col, "parsed_output", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate alignment metrics between topics and ground-truth topics."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/input/assignment.jsonl",
        help="Path to data file (containing both ground-truth and predicted topics)",
    )
    parser.add_argument(
        "--ground_truth_col",
        type=str,
        default="ground_truth",
        help="Column name for ground-truth topics",
    )
    parser.add_argument(
        "--output_col",
        type=str,
        default="output",
        help="Column name for predicted topics",
    )
    args = parser.parse_args()

    metric_calc(args.data_file, args.ground_truth_col, args.output_col)
