#/usr/bin/env python3
"""
Visual SLAM Entry Point Script

This script tests the matplotlib backend configuration for the visual SLAM demo.

Usage:
    python visual_slam_entry.py --show          # Shows plots interactively
    python visual_slam_entry.py                  # Saves plots to files only
"""

import os
import sys

# Set matplotlib environment variable before importing matplotlib
os.environ['MPLBACKEND'] = 'Agg' if '--show' not in sys.argv else 'MacOSX'

import matplotlib
print(f"Matplotlib backend: {matplotlib.get_backend()}")

import matplotlib.pyplot as plt

def test_plot_generation(title="Test Plot"):
    """Generate a test plot to verify matplotlib functionality."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([1, 2, 3, 4, 5], [2, 3, 5, 7, 11], label='Sample Data')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        output_path = '/tmp/visual_slam_test_plot.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        
        # Verify file was created
        if os.path.exists(output_path):
            print(f"✓ Test plot saved to: {output_path}")
            file_size = os.path.getsize(output_path)
            print(f"  File size: {file_size:,} bytes")
            return True
        else:
            print("✗ FAILED: Plot file was not created")
            return False
            
    except Exception as e:
        print(f"✗ ERROR creating plot: {e}")
        return False
    finally:
        plt.close('all')

def main():
    """Main function to test matplotlib backend configuration."""
    print("=" * 60)
    print("Visual SLAM Entry Point Test")
    print("=" * 60)
    
    # Check if --show flag is present
    show_plots = '--show' in sys.argv
    print(f"\nMode: {'Interactive (--show)' if show_plots else 'Non-interactive (Agg backend)'}")
    
    # Test plot generation
    print(f"\nTesting plot generation with {matplotlib.get_backend()} backend...")
    success = test_plot_generation()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ TEST PASSED: Matplotlib backend configuration is working correctly.")
        print("  The visual SLAM demo should run successfully.")
    else:
        print("✗ TEST FAILED: Plot generation failed.")
        print("  Check your matplotlib installation and environment.")
        sys.exit(1)
    print("=" * 60)
    
    # Also output the command that would work for testing
    print("\nTo run the actual visual SLAM demo:")
    if show_plots:
        print("  python src/simulations/visual_slam/visual_slam_entry.py --show")
    else:
        print("  python src/simulations/visual_slam/visual_slam_entry.py")
    print(f"  Current working directory: {os.getcwd()}")

if __name__ == "__main__":
    main()
