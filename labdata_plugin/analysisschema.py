from labdata.schema import get_user_schema  # type: ignore

chipmunkschema = get_user_schema()

# @chipmunkschema
# here I can start defining a stimulus selectivity class where i grab psth's from units and see if their response is more than 1.96 z's above baseline
