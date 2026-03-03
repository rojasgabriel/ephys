from labdata.schema import get_user_schema, prefs  # type: ignore
import datajoint as dj  # type: ignore

username = prefs["database"]["database.user"]
chipmunkschema = get_user_schema()  # allows user defined schemas
if "chipmunk_schema" in prefs.keys():  # to be able to override to another name
    chipmunkschema = prefs["chipmunk_schema"]
if type(chipmunkschema) is str:
    if "root" in chipmunkschema:
        raise (
            ValueError(
                '[chipmunk] "chipmunk_schema" must be specified in the preference file to run as root.'
            )
        )
    chipmunkschema = dj.schema(chipmunkschema)

# @chipmunkschema
# here I can start defining a stimulus selectivity class where i grab psth's from units and see if their response is more than 1.96 z's above baseline
