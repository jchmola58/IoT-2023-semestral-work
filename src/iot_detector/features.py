"""Feature list and simple validity rules.

Only these 39 columns are used as model inputs. All of them are computed
from the traffic itself, so they are known at prediction time.
`Label`, `y` and `source_file` are NEVER inputs (target / file identity).
"""
FEATURES = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
    "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Variance",
]

# Columns that leak the answer or identify the capture file.
TARGET_LIKE_COLUMNS = ["Label", "label", "y", "source_file"]

# 0/1 indicator columns
BINARY_FEATURES = [
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
    "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
]

# (low, high) allowed values. None = no limit on that side.
VALID_RANGES = {name: (0.0, None) for name in FEATURES}
VALID_RANGES.update({name: (0.0, 1.0) for name in BINARY_FEATURES})
VALID_RANGES["Protocol Type"] = (0.0, 255.0)
VALID_RANGES["Time_To_Live"] = (0.0, 255.0)

LABELS = {0: "BENIGN", 1: "MALICIOUS"}
