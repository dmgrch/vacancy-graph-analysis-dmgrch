import typing as tp

import requests  # type: ignore
from requests.adapters import HTTPAdapter  # type: ignore
from requests.packages.urllib3.util.retry import Retry  # type: ignore


class Session(requests.Session):
    """
    Сессия.

    :param base_url: Базовый адрес, на который будут выполняться запросы.
    :param timeout: Максимальное время ожидания ответа от сервера.
    :param max_retries: Максимальное число повторных запросов.
    :param backoff_factor: Коэффициент экспоненциального нарастания задержки.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        retries = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=True,
            raise_on_redirect=True,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.mount("https://", adapter)

    def _full_url(self, url: str) -> str:
        if not url:
            return self.base_url
        if url.startswith("https://"):
            return url
        return f"{self.base_url}/{url.lstrip('/')}"

    def get(self, url: str, *args: tp.Any, **kwargs: tp.Any) -> requests.Response:
        timeout = kwargs.pop("timeout", self.timeout)
        return super().get(self._full_url(url), timeout=timeout, **kwargs)

    def post(self, url: str, *args: tp.Any, **kwargs: tp.Any) -> requests.Response:
        timeout = kwargs.pop("timeout", self.timeout)
        return super().post(self._full_url(url), timeout=timeout, **kwargs)
