class AppendSlashMiddleware:
    """
    Silently appends a trailing slash to all requests that are missing one,
    before URL matching happens. This avoids Django's APPEND_SLASH redirect
    which breaks POST requests when a proxy strips trailing slashes.
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.path_info.endswith('/'):
            request.path_info = request.path_info + '/'
        return self.get_response(request)
