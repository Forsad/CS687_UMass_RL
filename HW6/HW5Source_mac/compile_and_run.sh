set -e
g++ -I. -Ofast  -std=c++11 ./*.cpp -o run_system
./run_system
python gridworld_error_bar.py
 
