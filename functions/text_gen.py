from google import genai
from google.genai import types

from .classes import Track
from crawl4ai import *
from googlesearch import search
import asyncio
import os
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)

# Keep for legacy callers in main.py
def get_llm():
    return _get_client()

# ----------------- CRAWLER LOGIC (FALLBACK IF GROUNDING QUOTA EXHAUSTED) -----------------
async def _crawl_url(url: str) -> str:
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        return result.markdown

def search_google(query: str, num_results: int = 1) -> list:
    results = []
    excluded_domains = [
        "youtube", "facebook", "genius", "bandcamp", "amazon", "musixmatch",
        "twitter", "instagram", "tiktok", "wikipedia", "reddit", "soundcloud",
        "dailymotion", "spotify", "apple", "discogs"
    ]
    try:
        web_data_array = list(search(query, num_results=num_results * 3, advanced=True))
        for web_data in web_data_array:
            try:
                if not any(domain in web_data.url for domain in excluded_domains):
                    results.append(web_data)
                    if len(results) >= num_results:
                        break
            except Exception:
                continue
    except Exception as e:
        print(f"[Crawler] Google search failed: {e}")

    # Also try to get a Wikipedia article for reliable facts
    try:
        wiki_data = list(search(f"{query} wikipedia", num_results=1, advanced=True))
        if wiki_data:
            results.insert(0, wiki_data[0])
    except Exception:
        pass

    return results[:num_results]

def process_search_results(search_results: list) -> str:
    raw_data = []
    for result in search_results:
        try:
            web_data = asyncio.run(_crawl_url(result.url))
            raw_data.append(web_data[:3000])  # Cap per-page to keep tokens low
        except Exception as e:
            print(f"[Crawler] Failed to crawl {result.url}: {e}")
            continue
    return "\n\n---\n\n".join(raw_data)

def transform_into_google_query(song: Track) -> str:
    return f"{song.name} by {song.artists}"
# -----------------------------------------------------------------------------------------

def radio_host_prompt(previous_song: Track, next_song: Track) -> str:
    return f"""
        Role: You're a veteran radio host with a warm, authoritative voice. Craft smooth 20-40 second transitions in English that:
            Respect the music
            Highlight meaningful context
            Flow naturally into the next track

        Create a smooth, conversational introduction that includes:
        - Previous song: {previous_song.name} by {previous_song.artists}
        - Next song: {next_song.name} by {next_song.artists} (from the album {next_song.album}, released in {next_song.release_year})

        Using Google Search (which is enabled for you), look up interesting facts, background story, production details, or lyrical meaning of the next song: "{next_song.name}" by {next_song.artists}.
        Include 1 or 2 of these real-world grounded facts in your transition.

        Style guidelines:
        - Use dynamic English with varied sentence structures.
        - Be conversational and engaging
        - Sound natural, not academic
        - Avoid cliché phrases like "musical journey" or "get ready"
        - Focus on concrete facts rather than vague statements

        IN ENGLISH!
        NEVER USE PHRASES LIKE "musical journey" OR "get ready"
        DONT TALK TOO MUCH ABOUT THE AMBIANCE OF THE SONG AND HOW IT MAKES YOU FEEL.
        ALWAYS PRESENT THE PREVIOUS SONG.
        200 WORDS MAXIMUM. ONLY TELL THE MOST PERTINENT INFORMATION.
        Never say you are a radio host. NEVER TALK ABOUT YOUR SHOW.
        DO NOT USE SYMBOLS LIKE: [, ], *, (, ), but quotes are ok.
    """

def radio_host_prompt_with_data(previous_song: Track, next_song: Track, web_data: str) -> str:
    return f"""
        Role: You're a veteran radio host with a warm, authoritative voice. Craft smooth 20-40 second transitions in English that:
            Respect the music
            Highlight meaningful context
            Flow naturally into the next track

        Create a smooth, conversational introduction that includes:
        - Previous song: {previous_song.name} by {previous_song.artists}
        - Next song: {next_song.name} by {next_song.artists} (from the album {next_song.album}, released in {next_song.release_year})

        Here is verified information about the next song or artist:
        {web_data}

        Include 1 or 2 of these real-world grounded facts in your transition.

        Style guidelines:
        - Use dynamic English with varied sentence structures.
        - Be conversational and engaging
        - Sound natural, not academic
        - Avoid cliché phrases like "musical journey" or "get ready"
        - Focus on concrete facts rather than vague statements

        IN ENGLISH!
        NEVER USE PHRASES LIKE "musical journey" OR "get ready"
        DONT TALK TOO MUCH ABOUT THE AMBIANCE OF THE SONG AND HOW IT MAKES YOU FEEL.
        ALWAYS PRESENT THE PREVIOUS SONG.
        200 WORDS MAXIMUM. ONLY TELL THE MOST PERTINENT INFORMATION.
        Never say you are a radio host. NEVER TALK ABOUT YOUR SHOW.
        DO NOT USE SYMBOLS LIKE: [, ], *, (, ), but quotes are ok.
    """

def generate_text(prompt: str, llm=None, use_grounding: bool = True) -> str:
    """
    Try each model in sequence. If use_grounding=True, attach the google_search tool
    (new SDK syntax). Falls back to the next model on any error.
    """
    # Ordered by preference: cheapest/fastest first
    models_to_try = [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]

    client = _get_client()
    last_error = None

    for model_name in models_to_try:
        try:
            print(f"[Gemini API] Attempting model: {model_name} (grounding={use_grounding})")

            config_kwargs = {}
            if use_grounding:
                config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

            config = types.GenerateContentConfig(**config_kwargs)

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            text = response.text
            if text and text.strip():
                print(f"[Gemini API] ✅ Success with {model_name} (grounding={use_grounding})")
                return text
            else:
                print(f"[Gemini API] ⚠️ {model_name} returned empty response.")

        except Exception as e:
            print(f"[Gemini API] ❌ Model {model_name} failed: {e}")
            last_error = e

    if last_error:
        raise last_error
    return ""


def generate_text_for_song(input: list, llm=None) -> str:
    full_text = ""
    if len(input) < 2:
        return full_text

    prev_track = input[0]
    next_track = input[1]

    # --- Attempt 1: Native Google Search Grounding (new SDK, google_search tool) ---
    try:
        prompt = radio_host_prompt(prev_track, next_track)
        full_text = generate_text(prompt, use_grounding=True)
        if full_text:
            print("[Strategy] ✅ Used native Google Search grounding.")
            return full_text
    except Exception as e:
        print(f"[Strategy] ❌ Native grounding failed: {e}. Falling back to Python web crawler...")

    # --- Attempt 2: Python crawler → feed as context, no grounding tool (free tier safe) ---
    try:
        query = transform_into_google_query(next_track)
        search_results = search_google(query, num_results=1)
        web_data = process_search_results(search_results)

        prompt_with_data = radio_host_prompt_with_data(prev_track, next_track, web_data)
        full_text = generate_text(prompt_with_data, use_grounding=False)
        if full_text:
            print("[Strategy] ✅ Used Python crawler fallback.")
            return full_text
    except Exception as e:
        print(f"[Strategy] ❌ Python crawler fallback failed: {e}")

    return full_text