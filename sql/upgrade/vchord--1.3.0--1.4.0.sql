-- List of functions

CREATE FUNCTION "_vchordrq_support_uuid_ops"() RETURNS TEXT /* String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_support_uuid_ops_wrapper';

CREATE FUNCTION "_vchordrq_support_int2_ops"() RETURNS TEXT /* String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_support_int2_ops_wrapper';

CREATE FUNCTION "_vchordrq_support_int4_ops"() RETURNS TEXT /* String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_support_int4_ops_wrapper';

CREATE FUNCTION "_vchordrq_support_int8_ops"() RETURNS TEXT /* String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_support_int8_ops_wrapper';

CREATE FUNCTION "_vchordrq_uuid_options"(
    "relopts" internal /* pgrx :: Internal */
) RETURNS void
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_uuid_options_wrapper';

CREATE FUNCTION "_vchordrq_int2_options"(
    "relopts" internal /* pgrx :: Internal */
) RETURNS void
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_int2_options_wrapper';

CREATE FUNCTION "_vchordrq_int4_options"(
    "relopts" internal /* pgrx :: Internal */
) RETURNS void
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_int4_options_wrapper';

CREATE FUNCTION "_vchordrq_int8_options"(
    "relopts" internal /* pgrx :: Internal */
) RETURNS void
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_int8_options_wrapper';

-- vchordrq_sampled_queries: also list indexes with a filter key column.
CREATE OR REPLACE FUNCTION vchordrq_sampled_queries(regclass)
RETURNS TABLE(
    schema_name NAME,
    index_name NAME,
    table_name NAME,
    column_name NAME,
    operator NAME,
    value TEXT
)
STRICT LANGUAGE plpgsql AS $$
DECLARE
    ext_schema TEXT;
    query_text TEXT;
BEGIN
    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension e
    JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';

    IF ext_schema IS NULL THEN
        RAISE EXCEPTION 'vchord is not installed';
    END IF;

    query_text := format(
        $q$
        WITH index_metadata AS (
            SELECT
                NS.nspname AS schema_name,
                I.relname AS index_name,
                C.relname AS table_name,
                PA.attname AS column_name,
                OP.oprname AS operator
            FROM
                pg_catalog.pg_index X
            JOIN
                pg_catalog.pg_class C ON C.oid = X.indrelid
            JOIN
                pg_catalog.pg_namespace NS ON C.relnamespace = NS.oid
            JOIN
                pg_catalog.pg_class I ON I.oid = X.indexrelid
            JOIN
                pg_catalog.pg_am A ON A.oid = I.relam
            LEFT JOIN
                pg_catalog.pg_opclass AS OPC ON OPC.oid = X.indclass[0]
            LEFT JOIN
                pg_catalog.pg_amop AO ON OPC.opcfamily = AO.amopfamily
            LEFT JOIN
                pg_catalog.pg_operator OP ON OP.oid = AO.amopopr
            LEFT JOIN
                pg_catalog.pg_attribute PA ON PA.attrelid = X.indrelid AND PA.attnum = X.indkey[0]
            WHERE
                A.amname = 'vchordrq'
                AND AO.amopstrategy = 1
                AND C.relkind = 'r'
                AND X.indexrelid = %1$s
        )
        SELECT
            im.schema_name,
            im.index_name,
            im.table_name,
            im.column_name,
            im.operator,
            s.value
        FROM
            index_metadata im,
            LATERAL %2$I.vchordrq_sampled_values(%1$s) AS s(value);
        $q$,
        $1::oid,
        ext_schema
    );
    RETURN QUERY EXECUTE query_text;
END;
$$;

-- List of operator families

CREATE OPERATOR FAMILY uuid_ops USING vchordrq;
CREATE OPERATOR FAMILY integer_ops USING vchordrq;

-- List of operator classes

-- Filter key columns: USING vchordrq (embedding ..., key [, key ...]), each with the
-- parameter `bytes` (support function 2): `key uuid_ops (bytes = 16)`. The first filter
-- key column orders the rows of each list and selects the blocks a scan reads; the others
-- are tested in the blocks it selects, and a scan that constrains only them reads every
-- block of the probed lists: put the most selective column first.

CREATE OPERATOR CLASS uuid_ops
    DEFAULT FOR TYPE uuid USING vchordrq FAMILY uuid_ops AS
    OPERATOR 1 = (uuid, uuid),
    FUNCTION 1 _vchordrq_support_uuid_ops(),
    FUNCTION 2 _vchordrq_uuid_options(internal);

CREATE OPERATOR CLASS int2_ops
    DEFAULT FOR TYPE int2 USING vchordrq FAMILY integer_ops AS
    OPERATOR 1 = (int2, int2),
    FUNCTION 1 _vchordrq_support_int2_ops(),
    FUNCTION 2 _vchordrq_int2_options(internal);

CREATE OPERATOR CLASS int4_ops
    DEFAULT FOR TYPE int4 USING vchordrq FAMILY integer_ops AS
    OPERATOR 1 = (int4, int4),
    FUNCTION 1 _vchordrq_support_int4_ops(),
    FUNCTION 2 _vchordrq_int4_options(internal);

CREATE OPERATOR CLASS int8_ops
    DEFAULT FOR TYPE int8 USING vchordrq FAMILY integer_ops AS
    OPERATOR 1 = (int8, int8),
    FUNCTION 1 _vchordrq_support_int8_ops(),
    FUNCTION 2 _vchordrq_int8_options(internal);

ALTER OPERATOR FAMILY integer_ops USING vchordrq ADD
    OPERATOR 1 = (int2, int4),
    OPERATOR 1 = (int2, int8),
    OPERATOR 1 = (int4, int2),
    OPERATOR 1 = (int4, int8),
    OPERATOR 1 = (int8, int2),
    OPERATOR 1 = (int8, int4);
