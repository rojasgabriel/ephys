import datajoint as dj
from labdata.schema import get_user_schema

rojasbowe_schema = get_user_schema()


@rojasbowe_schema
class EventMapping(dj.Manual):  # TODO: remove renaming of Digital keys
    definition = """
    -> Session
    event_name                           : varchar(54)   # shared logical event role
    ---
    -> DatasetEvents.Digital.proj(source_dataset_name='dataset_name', source_stream_name='stream_name', source_event_name='event_name')
    """


# @rojasbowe_schema
# class LocomotionPeaks(dj.Computed):
#     definition = """
#     -> UnitCount.Unit
#     ---
#     stat_peak       : float  # peak amplitude of stat event (sp/s)
#     stat_latency    : float  # latency of stat event (s)
#     move_peak       : float  # peak amplitude of move event (sp/s)
#     move_latency    : float  # latency of move event (s)
#     """

#     def make(self, key):
