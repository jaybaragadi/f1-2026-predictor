"""
FastF1 Live Timing Recorder
Records F1 sessions in real-time for immediate post-race analysis
"""

import fastf1
from fastf1 import livetiming
import argparse
from datetime import datetime
from pathlib import Path
import time

def record_session(year, race_round, session_type='R'):
    """
    Record a live F1 session
    
    Parameters:
    -----------
    year : int
        Season year (e.g., 2026)
    race_round : int
        Race round number (1-24)
    session_type : str
        'FP1', 'FP2', 'FP3', 'Q', 'S' (sprint), 'R' (race)
    """
    
    # Create directory for recordings
    recordings_dir = Path("live_recordings")
    recordings_dir.mkdir(exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = recordings_dir / f"{year}_R{race_round}_{session_type}_{timestamp}.jsonl"
    
    print(f"\n{'='*70}")
    print(f"🔴 LIVE TIMING RECORDER")
    print(f"{'='*70}")
    print(f"Year: {year}")
    print(f"Round: {race_round}")
    print(f"Session: {session_type}")
    print(f"Recording to: {filename}")
    print(f"{'='*70}\n")
    
    print("⏳ Waiting for session to start...")
    print("   Press Ctrl+C to stop recording\n")
    
    try:
        # Create live timing client
        client = livetiming.LiveTimingClient()
        
        # Open file for writing
        with open(filename, 'w') as f:
            # Start recording
            for message in client.messages():
                # Write each message to file
                f.write(message + '\n')
                f.flush()  # Ensure data is written immediately
                
                # Print status every 100 messages
                if client.message_count % 100 == 0:
                    print(f"✓ Recorded {client.message_count} messages...")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Recording stopped by user")
        print(f"✅ Saved to: {filename}")
        print(f"📊 Total messages: {client.message_count}")
        return filename
    
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        return None


def load_from_recording(recording_file):
    """
    Load a session from a saved recording
    
    Parameters:
    -----------
    recording_file : str or Path
        Path to the .jsonl recording file
    
    Returns:
    --------
    fastf1.core.Session
    """
    
    print(f"\n📂 Loading session from recording...")
    print(f"   File: {recording_file}")
    
    # Load the session from local file
    session = fastf1.core.Session.load_from_file(recording_file)
    
    print(f"✅ Session loaded successfully!")
    print(f"   Event: {session.event['EventName']}")
    print(f"   Session: {session.name}")
    print(f"   Laps: {len(session.laps)}")
    
    return session


def quick_analysis(recording_file):
    """
    Quick analysis of a recorded session
    """
    
    session = load_from_recording(recording_file)
    
    print(f"\n{'='*70}")
    print(f"📊 QUICK ANALYSIS")
    print(f"{'='*70}\n")
    
    # Get results
    results = session.results
    
    if results is not None and len(results) > 0:
        print("🏁 SESSION RESULTS:\n")
        for idx, row in results.head(10).iterrows():
            print(f"{int(row['Position']):2d}. {row['Abbreviation']:3s} - "
                  f"{row['FullName']:25s} ({row['TeamName']})")
        
        # Winner stats
        winner = results.iloc[0]
        print(f"\n🏆 WINNER: {winner['FullName']}")
        print(f"   Team: {winner['TeamName']}")
        print(f"   Time: {winner['Time']}")
    else:
        print("⚠️  No results available (session may still be in progress)")
    
    return session


def auto_record_race(year, race_round):
    """
    Automatically record qualifying and race for a Grand Prix weekend
    """
    
    print(f"\n{'='*70}")
    print(f"🎬 AUTO RACE RECORDER - {year} Round {race_round}")
    print(f"{'='*70}\n")
    
    print("This will record both Qualifying and Race sessions.")
    print("Make sure to start this script BEFORE qualifying begins!\n")
    
    response = input("Start recording? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Record qualifying
    print("\n📍 Step 1: Recording Qualifying Session")
    print("   Start this when Q1 begins...")
    input("   Press Enter when ready to record Qualifying...")
    quali_file = record_session(year, race_round, 'Q')
    
    if quali_file:
        print(f"\n✅ Qualifying recorded: {quali_file}")
    
    # Wait for race
    print("\n📍 Step 2: Recording Race Session")
    print("   Start this when the formation lap begins...")
    input("   Press Enter when ready to record Race...")
    race_file = record_session(year, race_round, 'R')
    
    if race_file:
        print(f"\n✅ Race recorded: {race_file}")
        
        # Offer quick analysis
        response = input("\nRun quick analysis? (y/n): ")
        if response.lower() == 'y':
            quick_analysis(race_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FastF1 Live Timing Recorder')
    
    parser.add_argument('--year', type=int, default=2026, help='Season year')
    parser.add_argument('--round', type=int, required=True, help='Race round (1-24)')
    parser.add_argument('--session', type=str, default='R', 
                       choices=['FP1', 'FP2', 'FP3', 'Q', 'S', 'R'],
                       help='Session type (R=Race, Q=Quali, S=Sprint)')
    parser.add_argument('--load', type=str, help='Load and analyze existing recording')
    parser.add_argument('--auto', action='store_true', help='Auto-record qualifying + race')
    
    args = parser.parse_args()
    
    if args.load:
        # Load existing recording
        quick_analysis(args.load)
    elif args.auto:
        # Auto-record weekend
        auto_record_race(args.year, args.round)
    else:
        # Record single session
        recording_file = record_session(args.year, args.round, args.session)
        
        if recording_file:
            response = input("\nRun quick analysis? (y/n): ")
            if response.lower() == 'y':
                quick_analysis(recording_file)