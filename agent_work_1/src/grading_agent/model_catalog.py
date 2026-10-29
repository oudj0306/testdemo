"""Vision-capable model presets for image-based grading."""

from __future__ import annotations

from urllib.parse import urlparse

MODEL_PROVIDERS = (
    {
        "id": "openai",
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ("gpt-5.4", "gpt-5.4-mini"),
        "docs_url": "https://developers.openai.com/api/docs/guides/images-vision",
        "key_url": "https://platform.openai.com/api-keys",
        "note": "支持图片输入和结构化输出；模型可用性取决于账号权限。",
    },
    {
        "id": "qwen",
        "name": "通义千问 · 百炼（北京）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ("qwen3.8-max", "qwen3.8-flash"),
        "docs_url": "https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-responses",
        "key_url": "https://help.aliyun.com/zh/model-studio/get-api-key",
        "note": "北京地域密钥可直接使用；其他地域或业务空间专属域名请用自定义连接。",
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "models": ("deepseek-flash",),
        "docs_url": "https://api-docs.deepseek.com/guides/responses_api/",
        "key_url": "https://platform.deepseek.com/api_keys",
        "note": "仅提供已确认支持图片输入的 deepseek-flash；文本模型不能直接识别试卷照片。",
    },
    {
        "id": "doubao",
        "name": "豆包 · 火山方舟",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "models": ("doubao-seed-2-0-pro-260215", "doubao-seed-2-0-lite-260215"),
        "docs_url": "https://console.volcengine.com/ark/region:cn-beijing/docs/create-model-responses-api",
        "key_url": "https://ark.volcengine.com/region:cn-beijing/apikey",
        "note": "北京地域方舟 API；需在方舟账号中开通所选模型。结构化输出仍标为 beta。",
    },
    {
        "id": "zhipu",
        "name": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/v1",
        "models": ("glm-5.3",),
        "docs_url": "https://docs.bigmodel.cn/cn/guide/develop/responses/introduction",
        "key_url": "https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys",
        "note": "使用智谱 Response API 地址；不是旧的 /api/paas/v4。",
    },
    {
        "id": "gemini",
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "models": ("gemini-3.8-flash",),
        "api_style": "chat",
        "chat_format": "json_schema",
        "docs_url": "https://ai.google.dev/gemini-api/docs/openai",
        "key_url": "https://aistudio.google.com/apikey",
        "note": "使用 Google 官方 OpenAI 兼容 Chat 接口；需确认所在地区及账号可用。",
    },
    {
        "id": "kimi",
        "name": "Kimi · 国际平台",
        "base_url": "https://api.moonshot.ai/v1",
        "models": ("kimi-k2.6", "kimi-k3"),
        "api_style": "chat",
        "chat_format": "json_object",
        "docs_url": "https://www.kimi.ai/help/kimi-api/api-overview",
        "key_url": "https://platform.kimi.ai/console/api-keys",
        "note": "国际平台密钥；K3 始终启用思考，K2.6 支持图像输入。国内平台密钥请使用对应地址。",
    },
    {
        "id": "hunyuan",
        "name": "腾讯混元 · 视觉",
        "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
        "models": ("hunyuan-vision",),
        "api_style": "chat",
        "docs_url": "https://cloud.tencent.com/document/product/1729/111007",
        "key_url": "https://console.cloud.tencent.com/hunyuan/api-key",
        "note": "使用混元视觉 Chat 接口；JSON 靠提示词与本地校验，正式批量使用前请用真实试卷验证。",
    },
    {
        "id": "baidu",
        "name": "百度千帆 · 国际平台",
        "base_url": "https://api.baiduqianfan.ai/v1",
        "models": ("ernie-5.0",),
        "api_style": "chat",
        "docs_url": "https://intl.cloud.baidu.com/en/doc/qianfan/s/7m95lyy43-intl-en",
        "key_url": "https://console.baiduqianfan.ai/",
        "note": "国际平台 ERNIE 5.0 视觉模型；国内千帆地址与密钥不同。JSON 靠提示词与本地校验。",
    },
    {
        "id": "anthropic",
        "name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "models": ("claude-sonnet-5-5", "claude-haiku-5-5"),
        "api_style": "anthropic",
        "docs_url": "https://platform.claude.com/docs/en/api/http/messages",
        "key_url": "https://console.anthropic.com/settings/keys",
        "note": "使用 Claude 原生 Messages 接口，支持图片与 JSON Schema；首次使用新结构化输出格式可能增加延迟。",
    },
)


def provider_for(base_url: str, model: str) -> str:
    normalized = base_url.rstrip("/")
    for provider in MODEL_PROVIDERS:
        if normalized == provider["base_url"] and model in provider["models"]:
            return provider["id"]
    return "custom"


def validate_model_connection(provider_id: str, model: str, base_url: str) -> tuple[str, str]:
    model = model.strip()
    base_url = base_url.strip().rstrip("/")
    provider = next((item for item in MODEL_PROVIDERS if item["id"] == provider_id), None)
    if provider_id != "custom":
        if provider is None or model not in provider["models"] or base_url != provider["base_url"]:
            raise ValueError("预设模型与 API 地址不匹配；如需其他型号或地址，请选自定义连接")
    parsed = urlparse(base_url)
    if (
        (
            parsed.scheme != "https"
            and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"})
        )
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "API 地址必须是 HTTPS 地址（本机 localhost 可用 HTTP），且不能包含账号、查询参数或片段"
        )
    return model, base_url


def apply_provider_options(payload: dict, base_url: str, enable_thinking: bool) -> None:
    """Translate the UI thinking switch to supported provider-specific fields."""
    host = urlparse(base_url).hostname or ""
    if host == "api.openai.com":
        payload["reasoning"] = {"effort": "high" if enable_thinking else "none"}
    if host == "api.deepseek.com":
        payload["reasoning"] = {"effort": "high" if enable_thinking else "none"}
    elif host.endswith(".volces.com"):
        payload["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}
    elif host == "open.bigmodel.cn":
        if enable_thinking:
            payload["reasoning"] = {"effort": "high"}
    else:
        # Preserve the pre-preset request shape for existing custom gateways,
        # including users' working Qwen-compatible proxy URLs.
        payload["enable_thinking"] = enable_thinking
