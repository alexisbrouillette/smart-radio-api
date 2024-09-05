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

def generate_text_for_song(input, llm): 
    #chat_response = user_proxy.initiate_chat(assistant, message=input, silent=False)
    final_prompt = f"""
        You are an energic music connoisseur. From the informations you receive, give me an 100 words presentation of the songs.
        If you know anything about the author, album or song, you can add it. Otherwise, don't.
        You are presenting the previous and the next song. Start with the previous one and end with the next one.
        Keep it short. MAXIMUM 100 words. IN FRENCH.

        Informations:
            previous song:
                {input[0]}
            next song:
                {input[1] if len(input) > 1 else 'no next song'}
        response: 
    """
    text = ""
    attempts = 0
    while len(text) == 0 and attempts < 3:
        text = llm.invoke(final_prompt)
        attempts += 1
    return text