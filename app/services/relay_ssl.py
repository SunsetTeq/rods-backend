import ssl


def build_ssl_context(
    verify: bool,
    ca_file: str = "",
) -> ssl.SSLContext:
    ca_file = ca_file.strip()
    if not verify:
        return ssl._create_unverified_context()

    if ca_file:
        return ssl.create_default_context(cafile=ca_file)

    return ssl.create_default_context()
