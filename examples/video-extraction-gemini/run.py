"""
Video Analysis with Gemini 2.5 Pro

This example demonstrates how to use Gemini 2.5 Pro with Instructor to analyze videos
and extract structured information. We'll process a video and extract tourist destinations
mentioned in it.

Requirements:
    pip install instructor google-genai pydantic

Usage:
    export GOOGLE_API_KEY=your_api_key_here
    
    python run.py path/to/your/video.mp4
"""

import instructor
from pydantic import BaseModel
import sys


class TouristDestination(BaseModel):
    """Represents a tourist destination mentioned in the video."""
    
    name: str
    description: str
    location: str


class VideoRecommendations(BaseModel):
    """Structured output containing recommendations from the video."""
    
    chain_of_thought: str
    description: str
    destinations: list[TouristDestination]


def analyze_video(video_path: str):
    """
    Analyze a video and extract tourist destination recommendations.
    
    Args:
        video_path: Path to the video file to analyze
        
    Returns:
        VideoRecommendations object containing structured data
    """
    client = instructor.from_provider(
        "google/gemini-2.0-flash-exp",
        async_client=False,
    )
    
    print(f"Uploading video: {video_path}")
    video = instructor.VideoWithGenaiFile.from_new_genai_file(video_path)
    print(f"Video uploaded successfully: {video.source}")
    
    print("Analyzing video content...")
    recommendations = client.messages.create(
        messages=[
            {
                "role": "user",
                "content": [
                    "What tourist destinations and places do they recommend in this video? "
                    "Provide a detailed analysis including the name, description, and location of each place.",
                    video,
                ],
            }
        ],
        response_model=VideoRecommendations,
    )
    
    return recommendations


def main():
    """Main function to run the video analysis."""
    if len(sys.argv) < 2:
        print("Usage: python run.py <path_to_video>")
        print("Example: python run.py travel_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    try:
        results = analyze_video(video_path)
        
        print("\n" + "=" * 80)
        print("VIDEO ANALYSIS RESULTS")
        print("=" * 80)
        
        print(f"\nOverview: {results.description}")
        print(f"\nAnalysis: {results.chain_of_thought}")
        
        print(f"\nDestinations Found: {len(results.destinations)}")
        print("-" * 80)
        
        for i, dest in enumerate(results.destinations, 1):
            print(f"\n{i}. {dest.name}")
            print(f"   Location: {dest.location}")
            print(f"   Description: {dest.description}")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"Error analyzing video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
