set -e
g++ -I. -Ofast  -std=c++11 ./*.cpp -o run_system
./run_system 0 0 0
python gridworld_error_bar.py
 
