import datajoint as dj
from labdata.schema import get_user_schema

rojasbowe_schema = get_user_schema()


@rojasbowe_schema
class EventMapping(dj.Manual):
    definition = """
    -> Session
    event_name                           : varchar(54)   # shared logical event role
    ---
    -> DatasetEvents.Digital.proj(source_dataset_name='dataset_name', source_stream_name='stream_name', source_event_name='event_name')
    """
