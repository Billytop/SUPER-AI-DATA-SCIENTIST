import argparse
import sys
import os
import io

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure backend directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # Try importing as module
    from reasoning.agents.surveyor_agent import run_surveyor
    from reasoning.agents.analyst_agent import run_analyst
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback: try direct import if we are inside the directory or weird path issues
    try:
        sys.path.append(os.path.join(current_dir, "reasoning", "agents"))
        import surveyor_agent
        import analyst_agent
        run_surveyor = surveyor_agent.run_surveyor
        run_analyst = analyst_agent.run_analyst
    except ImportError as e2:
        print(f"CRITICAL IMPORT ERROR: {e} | {e2}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="OmniBrain Autonomous Agents Runner")
    parser.add_argument('--mode', choices=['morning', 'night'], required=True, help="Which agent to run")
    
    args = parser.parse_args()
    
    if args.mode == 'morning':
        run_surveyor()
    elif args.mode == 'night':
        run_analyst()

if __name__ == "__main__":
    main()
