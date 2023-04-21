set -e
g++ -I. -Ofast  -std=c++11 ./*.cpp -o run_system
python evaluate_performance.py