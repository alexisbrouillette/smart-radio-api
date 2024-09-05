from pydantic import BaseModel


class Track(BaseModel):
    album: str
    name: str
    release_year: str
    artists: str
    id: str