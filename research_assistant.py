import asyncio
from crawl4ai import *
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from googlesearch import search

from classes import Track


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



def generate_text(input, llm, structure = None):
    text = ""
    attempts = 0
    while len(text) == 0 and attempts < 3:
        if structure:
            print("Calling LLM with structure")
            model_with_structure = llm.with_structured_output(structure)
            text = model_with_structure.invoke(input, structure=structure)
        else:
            text = llm.invoke(input)
        attempts += 1
    return text

def get_llm(): 
    return GoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key="AIzaSyA1kfa55giQ3yR5RxsbHXGl4-LboQ9_RHk",
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
)

async def main(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
        )
        return result.markdown
        
        # Save the result to a .txt file
        # with open("result.txt", "w", encoding="utf-8") as file:
        #     file.write(result.markdown)
    
def get_lyrics(query):
    lyrics_result = list(search(f"{query} lyrics", num_results=1, advanced=True))
    web_data =  asyncio.run(main(lyrics_result[0].url))
    lyrics = generate_text(lyrics_extractor_prompt(web_data), get_llm())
    return lyrics

def search_google(query, num_results=10):
    results = []
    web_data_array = list(search(query, num_results=num_results, advanced=True))
    
    wikipedia_data = list(search(f"{query} wikipedia", num_results=1, advanced=True))

    results.append(wikipedia_data[0])
    #if web_data countains ["youtube","facebook", ...]
    excluded_domains = ["youtube", "facebook","genius","bandcamp" "musixmatch", "twitter", "instagram", "tiktok", "wikipedia", "reddit", "soundcloud", "dailymotion", "spotify", "apple", "discogs"]
    for web_data in web_data_array:
        if not any(domain in web_data.url for domain in excluded_domains):
            results.append(web_data)
        else:
            print("Excluded: ", web_data.url)
    
    return results
def transform_into_google_query(song_infos: Track):
    query = f"{song_infos.name} by {song_infos.artists}"
    return query
def process_search_results(search_results):
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

if __name__ == "__main__":
    song_infos = Track(
        album="On the Lips",
        name="Crushed Velvet",
        release_year='2023',
        artists="Molly Lewis",
        id="1234567890",
    )
    query = transform_into_google_query(song_infos)
    
    llm = get_llm()

    lyrics = get_lyrics(query)

    song_google_search_results = search_google(query, 5)
    artist_google_search_results = search_google(f"{song_infos.artists} informations", 5)
    artist_album = f"{song_infos.artists} {song_infos.album} album informations"
    album_google_search_results = search_google(artist_album, 5)
    google_search_results = song_google_search_results + artist_google_search_results + album_google_search_results
    processed_results = process_search_results(google_search_results)
    print("Processed Results: ", processed_results)
    results = {
        "lyrics": lyrics,
        "google_search_results": processed_results
    }
    # keywords = [
    #     f"{query} collaboration",
    #     f"{query} creative process",
    #     f"{query} personal life",
    # ]
    # print("Extracted Text:")
    # print(extracted_text)
    # print("Keywords:")
    # print(keywords)
