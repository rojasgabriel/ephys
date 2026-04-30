import datajoint as dj
from labdata.schema import get_user_schema

rojasbowe_schema = get_user_schema()


@rojasbowe_schema
class SFN2026(dj.Manual):
    definition = """
    -> Subject
    """

    class Session(dj.Part):
        definition = """
        -> master
        -> Session
        """
