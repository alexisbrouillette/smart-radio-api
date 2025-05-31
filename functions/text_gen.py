from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory

from .classes import Track
from crawl4ai import *
from googlesearch import search
import asyncio


def get_llm(): 
    return GoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key="AIzaSyA1kfa55giQ3yR5RxsbHXGl4-LboQ9_RHk",
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
)
asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# def get_lyrics(song_infos: Track):
#     token = 'hHErmnc6rj857J0ryBwXna3_yCOm5CwfoqlUrrXU1HRBmVq8RBxNo8_wl2JLPXck'
#     genius = lyricsgenius.Genius(token)
#     # Use only the first artist if there are multiple separated by a comma
#     primary_artist = song_infos.artists.split(",")[0].strip()
#     song = genius.search_song(song_infos.name, primary_artist)
#     if song is not None:
#         lyrics = song.lyrics
#         return lyrics
#     else:
#         return "No lyrics found for this song."

def generate_prompt(song_infos: Track):
    prompt = f"""
        Song: {song_infos.name}	
        Album: {song_infos.album}
        Author: {song_infos.artists}
        Year: {song_infos.release_year}
    """
    return prompt


def researcher_prompt (content: str):
    return f"""
    Role: You are a data extraction agent for a radio show. Your task is to scan web content (articles, reviews, social media) and retrieve only the most radio-worthy information. Ignore fluff (menus, footers).
    Instructions:
        Identify the core topic:
            Song/album name, artists, release date.

    Content type (e.g., "review," "news scoop," "interview").

    Radio worthy details:

        How it connects to the artist's career.

        How it was made (e.g., "recorded in a bathroom").
        
        How it relates to the artist's personal life.

        Drama/Conflict: Feuds, clapbacks, legal issues.

        Surprises: Unplanned drops, secret features, weird production (e.g., "recorded in a bathroom").

    PLEASE GO INTO DETAILS.

    Attribute claims (e.g., "Pitchfork claims: [quote]").

    DO NOT ADD ANYTHING ELSE.
    ONLY RETURN YOUR FINDINGS.
    DO not talk about the structure of the website
    DO NOT MAKE UP FACTS. If you don't know, skip it.
    Here is the content you need to analyze:
    {content}
"""

def radio_host_prompt(previous_song: Track, next_song: Track, lyrics: str, web_data: str):
    prompt = f"""
        Role: You're a veteran radio host with a warm, authoritative voice. Craft smooth 20-40 second transitions that:
            Respect the music
            Highlight meaningful context
            Flow naturally into the next track

        Create a smooth, conversational introduction that includes:
        - Previous song presentation
        - Next song:
            - Song title, artist, year, and album
            - A presentation of the artist and album
            - A presentation of the song
            - 1-2 most interesting facts from the analysis
            - A natural transition to playing the song

        Style guidelines:
        - **Use dynamic French with varied sentence structures**
        - Be conversational and engaging
        - Sound natural, not academic
        - **Avoid cliché phrases like "voyage musical" or "préparez-vous"**
        - **Focus on concrete facts rather than vague statements**

        Example Outputs:
            1. Artist Context Focus:
            "We've just heard Joni Mitchell's 'Blue' - a record that redefined confessional songwriting. Now, we turn to another artist who transformed pain into poetry. Fiona Apple recorded 'Shadowboxer' at just 17 years old, grappling with the early pressures of fame. Listen for that devastating line 'You made me a shadowboxer, baby' - the way her voice breaks still gives me chills. Here's Fiona Apple."

            2. Production Focus:
            "That was D'Angelo's 'Untitled (How Does It Feel)' - pure soul perfection. Next up: Frank Ocean's 'Nikes'. What many don't know is that the warped vocals were created by running the track through an old SP-303 sampler, then re-recording it to tape. That ghostly quality you hear? Absolute studio alchemy. Listen closely as we go into 'Nikes' now."

            3. Substantive Drama Angle:
            "We just played Taylor Swift's 'My Tears Ricochet' from her pandemic-era work. Now, here's something equally poignant - Phoebe Bridgers' 'Kyoto', written about her complicated relationship with her father while touring Japan. The horns you'll hear were actually a last-minute addition after the band heard a mariachi group outside their studio. Let's go to 'Kyoto'."

        IN FRENCH!
        **NEVER USE PHRASES LIKE "PRÉPAREZ-VOUS" OR "VOYAGE MUSICAL"**
        DONT TALK TOO MUCH ABOUT THE AMBIANCE OF THE SONG AND HOW IT MAKES YOU FEEL.
        ALWAYS PRESENT THE PREVIOUS SONG.
        200 WORDS MAXIMUM. ONLY TELL THE MOST PERTINENT INFORMATION.
        Never say you are a radio host. NEVER TALK ABOUT YOUR SHOW.
        FOCUS ON THE INFORMATIONS COMING FROM THE Next song informations. Those informations are verified and true.
        ALWAYS DEVELOP WHAT YOU ARE SAYING. DO NOT JUST LIST THE FACTS.
        DO NO USE SYMBOLS LIKE: [, ], *, (,), but quotes are ok.

        Previous Song: {previous_song}
        Next Song: {next_song}
        Song lyrics: {lyrics}
        Next song informations: {web_data}
    """
    return prompt


def lyrics_extractor_prompt(lyrics: str):
    return f"""
    Role: Your task is to extract the lyrics of a song from a web page.
    You will receive a web page in a markdown format.
    You need to return ONLY the lyrics of the song.
    DO NOT ADD ANYTHING ELSE.
    If there are no lyrics in the web page, return "No lyrics found".
    Here is the content you need to analyze:
    {lyrics}
    """


def generate_text(input, llm):
    text = ""
    attempts = 0
    while len(text) == 0 and attempts < 3:
        text = llm.invoke(input)
        attempts += 1
    return text

async def main(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
        )
        return result.markdown
        
        # Save the result to a .txt file
        # with open("result.txt", "w", encoding="utf-8") as file:
        #     file.write(result.markdown)
    
def get_lyrics(query, llm):
    lyrics_result = list(search(f"{query} lyrics", num_results=1, advanced=True))
    web_data =  asyncio.run(main(lyrics_result[0].url))
    lyrics = generate_text(lyrics_extractor_prompt(web_data), llm)
    return lyrics

def search_google(query, num_results=10):
    results = []
    web_data_array = list(search(query, num_results=num_results, advanced=True))
    
    wikipedia_data = list(search(f"{query} wikipedia", num_results=1, advanced=True))

    results.append(wikipedia_data[0])
    #if web_data countains ["youtube","facebook", ...]
    excluded_domains = ["youtube", "facebook","genius","bandcamp", "amazon", "musixmatch", "twitter", "instagram", "tiktok", "wikipedia", "reddit", "soundcloud", "dailymotion", "spotify", "apple", "discogs"]
    for web_data in web_data_array:
        if not any(domain in web_data.url for domain in excluded_domains):
            results.append(web_data)
    
    return results

def transform_into_google_query(song_infos: Track):
    query = f"{song_infos.name} by {song_infos.artists}"
    return query
def process_search_results(search_results, llm):
    scraped_urls = []
    raw_data = []
    for result in search_results:
        try:
            if result.url in scraped_urls:
                print("Already scraped: ", result.url)
                continue
            web_data = asyncio.run(main(result.url))
            raw_data.append(web_data)
            scraped_urls.append(result.url)
        except Exception as e:
            print("Error processing search result: ", e)
            continue
        #wikipedia_extracted = generate_text(researcher_prompt(web_data), llm)
    extracted_text = generate_text(researcher_prompt(raw_data), llm)
    return extracted_text

def generate_text_for_song(input: list[Track], llm):
    full_text = ""
    # lyrics = get_lyrics(input[1])
    if(len(input) > 1):
        # historian_text = generate_text(historian_prompt(input[1], lyrics), llm)
        # print(historian_text)
        # researcher_text = generate_text(researcher_prompt(input[1], lyrics), llm)
        # print(researcher_text)
        # full_text = generate_text(radio_host_prompt(input[0], input[1], researcher_text, historian_text, lyrics), llm)
        query = transform_into_google_query(input[1])
        lyrics = get_lyrics(query, llm)
        song_google_search_results = search_google(query, 5)
        artist_google_search_results = search_google(f"{input[1].artists} informations", 5)
        artist_album = f"{input[1].artists} {input[1].album} album informations"
        album_google_search_results = search_google(artist_album, 5)
        google_search_results = song_google_search_results + artist_google_search_results + album_google_search_results
        processed_results = process_search_results(google_search_results, llm)
        full_text = generate_text(radio_host_prompt(input[0], input[1], lyrics, processed_results), llm)
        print("Full text: ", full_text)
    return full_text