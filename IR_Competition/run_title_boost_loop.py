import os
import subprocess

TEMPLATE_FILE = "boost_template.xml"
QRELS_FILE = "/data/IRCompetition/qrels_50_Queries"
boost_values = [0.1, 0.2, 0.3, 0.4, 0.5]

os.makedirs("boost_results", exist_ok=True)

for boost in boost_values:
    body = round(1.0 - boost, 2)
    boost_str = f"{boost:.1f}"
    body_str = f"{body:.1f}"

    param_file = f"boost_results/boost_title_{boost_str}.param"
    res_file = f"boost_results/run_boost_title_{boost_str}.res"

    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # ✅ Sécurité contre mauvaises substitutions
    content = content.replace("INSERT_BODY", body_str).replace("INSERT_BOOST", boost_str)

    with open(param_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"▶️ Running boost title = {boost_str}...")
    result = subprocess.run([
        "/home/student/indri-5.8-install/bin/IndriRunQuery", param_file
    ], stdout=open(res_file, "w"))

    if result.returncode != 0:
        print(f"❌ IndriRunQuery failed for boost = {boost_str}")
        continue

    print(f"📊 MAP for title boost {boost_str}:")
    subprocess.run([
        "/home/student/trec_eval-9.0.7/trec_eval", "-c", "-m", "map",
        QRELS_FILE, res_file
    ])
    print("-------------------------------------------")

