from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory

from .classes import Track

def get_llm(): 
    return GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key="AIzaSyApQSQyEx3zKjdAvg0C6GmFk1OuxY7PUgQ",
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
)

def generate_prompt(song_infos: Track):
    prompt = f"""
        Song: {song_infos.name}	
        Album: {song_infos.album}
        Author: {song_infos.artists}
        Year: {song_infos.release_year}
    """
    return prompt


character_prompt = "You are amusic enthusiast and the host of a music talk show. NEVER TALK ABOUT YOUR SHOW."

def historian_prompt(song_infos: Track):
    prompt = f"""
        You are a music historian analyzing historical context. Cover:
        - Historical events during the song's creation
        - Social and cultural movements of the era
        - How history influenced the song's meaning
        DO NOT MAKE UP FACTS. If you don't know, skip it.

        Song informations: {song_infos}
    """
    return prompt

def researcher_prompt(song_infos: Track):
    prompt = f"""
        You are a music researcher analyzing the song's background. Cover:
        - Artist's early career
        - Influences on the songwriting process
        - The song's impact on the artist's career
        - The lyrics and music composition

        DO NOT MAKE UP FACTS. If you don't know, skip it.

        Song informations: {song_infos}
    """
    return prompt

def radio_host_prompt(previous_song: Track, next_song: Track, researcher_analysis: str, historian_analysis: str):
    prompt = f"""
        You are a radio DJ creating the transition between songs. You will present the previous song, but most importantly, introduce the next song.
    Review the completed analysis and create an engaging introduction. Use the analysis from the researcher and historian to craft a compelling narrative for the next song.
    
    Create a smooth, conversational introduction that includes:
       - Song title, artist, year, and album
       - 1-2 most interesting facts from the analysis
       - A natural transition to playing the song

    Style guidelines:
    - Be conversational and engaging
    - Sound natural, not academic
    - Create excitement for the song
    - Use a smooth transition to introduce the song

    Example: "That was [previous song]. Now, let me take you back to 1965 with one of The Beatles' most beloved classics. 'Yesterday', from their album 'Help!', started as a dream in Paul McCartney's head - he literally woke up with the melody! This heartfelt ballad about loss and regret would become the most covered song in popular music history. Let's take a moment to appreciate this timeless masterpiece..."


    IN FRENCH!
    DONT TALK TOO MUCH ABOUT THE AMBIANCE OF THE SONG AND HOW IT MAKES YOU FEEL.
    USE THE ANALYSIS FROM THE RESEARCHER AND HISTORIAN TO CRAFT A COMPELLING NARRATIVE FOR THE NEXT SONG.
    ALWAYS PRESENT THE PREVIOUS SONG.
    100 WORDS MAXIMUM. ONLY TELL THE MOST PERTINENT INFORMATION.

    Never say you are a radio host. NEVER TALK ABOUT YOUR SHOW.
    
    Researcher analysis: {researcher_analysis}
    Historian analysis: {historian_analysis}


    Previous Song: {previous_song}
    Next Song: {next_song}

    """
    return prompt


def generate_text(input, llm):
    text = ""
    attempts = 0
    while len(text) == 0 and attempts < 3:
        text = llm.invoke(input)
        attempts += 1
    return text

def generate_text_for_song(input, llm):
    full_text = ""

    if(len(input) > 1):
        historian_text = generate_text(historian_prompt(input[1]), llm)
        print(historian_text)
        researcher_text = generate_text(researcher_prompt(input[1]), llm)
        print(researcher_text)
        full_text = generate_text(radio_host_prompt(input[0], input[1], researcher_text, historian_text), llm)


    return full_text