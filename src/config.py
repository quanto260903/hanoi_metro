"""
Hanoi Metro AI Agent - Central Configuration Module.
Tải cấu hình từ file .env theo chuẩn OpenAI-compatible API cho VLM, LLM, Embedding, và Qdrant.
Hỗ trợ chuyển đổi liền mạch giữa Cloud API và Local Host (Ollama / vLLM).
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

# Nạp file .env từ thư mục gốc dự án
try:
    from dotenv import load_dotenv
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _ENV_FILE = _PROJECT_ROOT / ".env"
    if _ENV_FILE.exists():
        load_dotenv(dotenv_path=_ENV_FILE)
    else:
        load_dotenv()
except ImportError:
    pass


class VLMConfig(BaseModel):
    """Cấu hình kết nối Vision Language Model (Tự host Ollama hoặc OpenAI-compatible API)."""
    provider: str = Field(
        default_factory=lambda: os.getenv("VLM_PROVIDER", "ollama"),
        description="Nhà cung cấp VLM: 'ollama' (tự host) hoặc 'openai'"
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv("VLM_BASE_URL", "http://127.0.0.1:11434"),
        description="Base URL cho VLM (Mặc định: http://localhost:11434 của Ollama)"
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("VLM_API_KEY", "ollama"),
        description="API Key cho VLM"
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("VLM_MODEL_NAME", "qwen2.5vl:3b"),
        description="Tên mô hình VLM tự host trên Ollama"
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("VLM_TEMPERATURE", "0.1")),
        description="Nhiệt độ sinh text (thấp để trích xuất dữ liệu chính xác)"
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("VLM_MAX_TOKENS", "16384")),
        description="Số token tối đa cho kết quả trả về của VLM (trang bảng rất dày có thể cần tới 16.384)"
    )
    timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("VLM_TIMEOUT_SECONDS", "180")),
        description="Thời gian timeout khi gọi VLM tự host"
    )
    num_ctx: int = Field(
        default_factory=lambda: int(os.getenv("VLM_NUM_CTX", "6144")),
        description="Context window Ollama cấp cho model (mặc định 16384 tokens)"
    )

    budget_usd: float = Field(
        default_factory=lambda: float(os.getenv("VLM_BUDGET_USD", "1.0")),
        description="Hạn mức chi phí VLM API đã duyệt (USD, cộng dồn mọi lần chạy); chạm hạn mức thì pipeline dừng"
    )
    strips: int = Field(
        default_factory=lambda: int(os.getenv("VLM_STRIPS", "1")),
        description="Số dải ngang cắt mỗi trang gửi VLM API (1 = cả trang một ảnh, dùng cho OpenAI; Gemini: 3)"
    )
    image_detail: str = Field(
        default_factory=lambda: os.getenv("VLM_IMAGE_DETAIL", "high"),
        description="'high' cho OpenAI (giữ độ phân giải ảnh); đặt trống khi dùng Gemini"
    )
    batch_discount: float = Field(
        default_factory=lambda: float(os.getenv("VLM_BATCH_DISCOUNT", "0.5")),
        description="Hệ số giá Batch API so với gọi thường (OpenAI, Gemini: 0.5)"
    )
    # Giá tham khảo gpt-6-luna (USD/1M token) – đối chiếu bảng giá chính thức rồi đặt lại trong .env
    price_in_per_m: float = Field(default_factory=lambda: float(os.getenv("VLM_PRICE_IN_PER_M", "0.10")))
    price_out_per_m: float = Field(default_factory=lambda: float(os.getenv("VLM_PRICE_OUT_PER_M", "0.50")))
    reasoning_effort: str = Field(
        default_factory=lambda: os.getenv("VLM_REASONING_EFFORT", ""),
        description="Trống = không gửi tham số; 'none' tắt thinking khi dùng Gemini 2.5 Flash (token suy luận tính theo giá đầu ra)"
    )

    @property
    def is_ollama(self) -> bool:
        """Kiểm tra xem VLM có đang trỏ về Ollama self-hosted hay không."""
        return self.provider.lower() == "ollama" or "11434" in self.base_url

    @property
    def is_cloud(self) -> bool:
        """VLM qua API đám mây (Gemini / OpenAI-compatible): dùng CloudVLMParser có hạn mức và kiểm tra tài liệu mật."""
        return self.provider.lower() in ("gemini", "cloud", "openai")


class LLMConfig(BaseModel):
    """Cấu hình kết nối Large Language Model cho tác vụ lập báo cáo và reasoning."""
    base_url: str = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "ollama")
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL_NAME", "qwen2.5:14b")
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "6144"))
    )
    timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
    )


class EmbeddingConfig(BaseModel):
    """Cấu hình kết nối Embedding Model để vector hóa tài liệu."""
    base_url: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:11434/v1")
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_API_KEY", "ollama")
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "bge-m3")
    )
    dimension: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    )
    timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "45"))
    )


class QdrantConfig(BaseModel):
    """Cấu hình kết nối cơ sở dữ liệu vector Qdrant."""
    host: str = Field(default_factory=lambda: os.getenv("QDRANT_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    grpc_port: int = Field(default_factory=lambda: int(os.getenv("QDRANT_GRPC_PORT", "6334")))
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY", "") or None)
    collection_name: str = Field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION_NAME", "hanoi_metro_documents")
    )

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class IngestionConfig(BaseModel):
    """Cấu hình thư mục lưu trữ và ngưỡng phân loại trang của Ingestion Pipeline."""
    raw_pdfs_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("RAW_PDFS_DIR", "data/raw_pdfs"))
    )
    processed_md_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("PROCESSED_MD_DIR", "data/processed_md"))
    )
    state_db_path: Path = Field(
        default_factory=lambda: Path(os.getenv("STATE_DB_PATH", "data/state.db"))
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # Các ngưỡng phân loại trang (Page Classifier)
    image_dpi: int = Field(
        default_factory=lambda: int(os.getenv("IMAGE_DPI", "300")),
        description="Độ phân giải render ảnh PDF cho Classifier (chuẩn 300 DPI)"
    )
    image_dpi_vlm: int = Field(
        default_factory=lambda: int(os.getenv("IMAGE_DPI_VLM", "150")),
        description="Độ phân giải render ảnh PDF gửi cho VLM API (150 DPI, detail=high với OpenAI)"
    )
    image_ratio_threshold: float = Field(
        default_factory=lambda: float(os.getenv("IMAGE_RATIO_THRESHOLD", "0.15")),
        description="Ngưỡng diện tích ảnh tối đa cho luồng DIRECT_TEXT"
    )
    classifier_blur_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CLASSIFIER_BLUR_THRESHOLD", "90.0")),
        description="Phương sai toán tử Laplacian; nhỏ hơn ngưỡng này xem như bị mờ"
    )
    classifier_skew_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CLASSIFIER_SKEW_THRESHOLD", "1.5")),
        description="Độ lệch góc nghiêng tối đa cho phép (độ)"
    )
    min_text_chars_direct: int = Field(
        default_factory=lambda: int(os.getenv("MIN_TEXT_CHARS_DIRECT", "120")),
        description="Số ký tự tối thiểu để xác nhận có lớp text chuẩn"
    )
    max_font_error_ratio: float = Field(
        default_factory=lambda: float(os.getenv("MAX_FONT_ERROR_RATIO", "0.03")),
        description="Tỉ lệ ký tự lỗi font tối đa chấp nhận được"
    )
    complex_table_cell_threshold: int = Field(
        default_factory=lambda: int(os.getenv("COMPLEX_TABLE_CELL_THRESHOLD", "4")),
        description="Số ô bảng lưới tối thiểu để đánh dấu là bảng phức tạp"
    )


class ChunkingConfig(BaseModel):
    """Tham số chunking theo hồ sơ (Mục 4.4 đề xuất); đơn vị là token ước lượng."""
    h1_single_max: int = Field(default_factory=lambda: int(os.getenv("CHUNK_H1_SINGLE_MAX", "1000")))
    child_min: int = Field(default_factory=lambda: int(os.getenv("CHUNK_CHILD_MIN", "250")))
    child_max: int = Field(default_factory=lambda: int(os.getenv("CHUNK_CHILD_MAX", "500")))
    parent_max: int = Field(default_factory=lambda: int(os.getenv("CHUNK_PARENT_MAX", "2000")))
    overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "60")))
    h2_article_max: int = Field(default_factory=lambda: int(os.getenv("CHUNK_H2_ARTICLE_MAX", "800")))
    table_single_max: int = Field(default_factory=lambda: int(os.getenv("CHUNK_TABLE_SINGLE_MAX", "600")))
    table_rows: int = Field(default_factory=lambda: int(os.getenv("CHUNK_TABLE_ROWS", "20")))


class RetrievalConfig(BaseModel):
    """Cấu hình tìm kiếm lai 2 nhánh (Qdrant + từ khóa) và kho văn bản."""
    keyword_backend: str = Field(
        default_factory=lambda: os.getenv("KEYWORD_BACKEND", "sqlite"),
        description="'sqlite' (thử nghiệm, một máy) hoặc 'postgres' (triển khai)"
    )
    sqlite_path: Path = Field(default_factory=lambda: Path(os.getenv("SQLITE_INDEX_PATH", "data/index.sqlite")))
    postgres_dsn: str = Field(default_factory=lambda: os.getenv("POSTGRES_DSN", ""))
    chunks_collection: str = Field(default_factory=lambda: os.getenv("QDRANT_CHUNKS_COLLECTION", "hanoi_metro_chunks"))
    qdrant_local_path: Path = Field(
        default_factory=lambda: Path(os.getenv("QDRANT_LOCAL_PATH", "data/qdrant_local")),
        description="Qdrant nhúng khi không kết nối được Qdrant server (chỉ một tiến trình được mở)"
    )
    synonym_dict_path: Path = Field(default_factory=lambda: Path(os.getenv("SYNONYM_DICT_PATH", "data/dictionaries/synonym_dict.json")))
    blocklist_dir: Optional[Path] = Field(
        default_factory=lambda: Path(os.getenv("BLOCKLIST_DIR")) if os.getenv("BLOCKLIST_DIR") else None,
        description="Thư mục kết quả page_router chứa restricted_md5.txt, blocklist_md5.txt, allowlist_md5.txt"
    )
    rrf_k: int = Field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    branch_k: int = Field(default_factory=lambda: int(os.getenv("SEARCH_BRANCH_K", "50")))
    max_chunks_per_doc: int = Field(default_factory=lambda: int(os.getenv("SEARCH_MAX_CHUNKS_PER_DOC", "3")))
    final_k: int = Field(default_factory=lambda: int(os.getenv("SEARCH_FINAL_K", "8")))
    parent_expand: int = Field(default_factory=lambda: int(os.getenv("SEARCH_PARENT_EXPAND", "6")))
    rerank: str = Field(
        default_factory=lambda: os.getenv("RERANK", "auto"),
        description="'auto' (chỉ bật khi có bge-reranker-v2-m3 qua FlagEmbedding), 'on', 'off'"
    )


class AgentConfig(BaseModel):
    """Cấu hình trợ lý (hỏi đáp, tóm tắt, lập báo cáo) và MCP server cho LibreChat."""
    mcp_host: str = Field(default_factory=lambda: os.getenv("MCP_HOST", "127.0.0.1"))
    mcp_port: int = Field(default_factory=lambda: int(os.getenv("MCP_PORT", "8765")))
    gateway_host: str = Field(default_factory=lambda: os.getenv("GATEWAY_HOST", "127.0.0.1"))
    gateway_port: int = Field(default_factory=lambda: int(os.getenv("GATEWAY_PORT", "8766")))
    gateway_api_key: str = Field(
        default_factory=lambda: os.getenv("GATEWAY_API_KEY", ""),
        description="Khóa LibreChat gửi trong header Authorization khi gọi dịch vụ hỏi đáp 1 lượt"
    )
    doc_link_base_url: str = Field(
        default_factory=lambda: os.getenv("DOC_LINK_BASE_URL", ""),
        description="Địa chỉ trình duyệt người dùng dùng để mở tài liệu gốc qua gateway; trống = http://<GATEWAY_HOST>:<GATEWAY_PORT>"
    )
    doc_link_secret: str = Field(
        default_factory=lambda: os.getenv("DOC_LINK_SECRET", "") or os.getenv("GATEWAY_API_KEY", ""),
        description="Khóa ký liên kết mở tài liệu gốc (mặc định suy ra từ GATEWAY_API_KEY); trống = tắt liên kết"
    )
    doc_roots: str = Field(
        default_factory=lambda: os.getenv("DOC_ROOTS", "D:/40GB;data/raw_pdfs"),
        description="Các thư mục (cách nhau bởi ;) chứa file gốc được phép mở từ trích dẫn"
    )
    access_control: bool = Field(
        default_factory=lambda: os.getenv("ACCESS_CONTROL", "off").strip().lower() in ("1", "on", "true", "bat"),
        description="Bật phân quyền theo phòng ban (src.access). Tắt = mọi người dùng xem toàn bộ kho như trước"
    )
    access_config_dir: str = Field(
        default_factory=lambda: os.getenv("ACCESS_CONFIG_DIR", "config/access"),
        description="Thư mục chứa phong_ban.yaml, vai_tro.yaml, nguoi_dung.csv, cap_rieng.csv"
    )
    access_user_header: str = Field(
        default_factory=lambda: os.getenv("ACCESS_USER_HEADER", "x-user-email"),
        description="Header LibreChat gửi email người dùng ({{LIBRECHAT_USER_EMAIL}}); chỉ tin khi kèm đúng GATEWAY_API_KEY"
    )
    access_log: str = Field(
        default_factory=lambda: os.getenv("ACCESS_LOG", "data/access_log.jsonl"),
        description="Nhật ký truy cập: ai, chức năng nào, được trả những văn bản nào (không ghi nội dung câu hỏi)"
    )
    access_registrations: str = Field(
        default_factory=lambda: os.getenv("ACCESS_REGISTRATIONS", "data/access/dang_ky.csv"),
        description="Sổ khai báo phòng ban (/phongban) và kết quả duyệt; người được duyệt được ghi vào nguoi_dung.csv"
    )
    doc_link_ttl_hours: float = Field(
        default_factory=lambda: float(os.getenv("DOC_LINK_TTL_HOURS", "24")),
        description="Khi bật phân quyền: liên kết mở file gắn với người dùng và hết hạn sau số giờ này (0 = không hết hạn)"
    )
    low_read_ratio: float = Field(
        default_factory=lambda: float(os.getenv("ENRICH_LOW_READ_RATIO", "0.8")),
        description="Văn bản có tỷ lệ trang đọc được dưới ngưỡng này: câu trả lời kèm ghi chú nên xem bản gốc"
    )
    enrich_results_dir: str = Field(
        default_factory=lambda: os.getenv("CORPUS_RESULTS_DIR", r"D:\40GB\_phan_tich_chi_phi_VLM\ket_qua"),
        description="Thư mục kết quả page_router (pages_routed.csv) dùng để tính tỷ lệ trang scan"
    )
    enrich_ai_budget_usd: float = Field(
        default_factory=lambda: float(os.getenv("ENRICH_AI_BUDGET_USD", "1.0")),
        description="Hạn mức chi phí (USD, cộng dồn mọi lần chạy) cho bước AI của src.indexing.enrich --ai; riêng với VLM_BUDGET_USD"
    )
    enrich_price_in_per_m: float = Field(default_factory=lambda: float(os.getenv("ENRICH_PRICE_IN_PER_M", "0.10")))
    enrich_price_out_per_m: float = Field(default_factory=lambda: float(os.getenv("ENRICH_PRICE_OUT_PER_M", "0.50")))
    enrich_ai_ledger: str = Field(
        default_factory=lambda: os.getenv("ENRICH_AI_LEDGER", "data/enrich_ai_ledger.sqlite"),
        description="Sổ ghi các lượt gọi AI của bước bổ sung metadata (mỗi văn bản một lần; căn cứ hạn mức)"
    )
    mcp_allowed_hosts: str = Field(
        default_factory=lambda: os.getenv("MCP_ALLOWED_HOSTS", "127.0.0.1:*,localhost:*,host.docker.internal:*"),
        description="Header Host được chấp nhận (chống DNS rebinding); LibreChat trong Docker gọi qua host.docker.internal"
    )
    context_budget_tokens: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_CONTEXT_BUDGET_TOKENS", "6000")),
        description="Tổng token ngữ cảnh tài liệu đưa vào một lần gọi LLM"
    )
    summary_batch_tokens: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_SUMMARY_BATCH_TOKENS", "3000")),
        description="Kích thước mỗi phần khi tóm tắt văn bản dài theo kiểu map-reduce"
    )
    organization: str = Field(default_factory=lambda: os.getenv("AGENT_ORGANIZATION", "CÔNG TY TNHH MTV ĐƯỜNG SẮT HÀ NỘI"))
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.1")),
        description="Nhiệt độ LLM cho hỏi đáp / tóm tắt / soạn thảo (thấp để bám sát tài liệu)"
    )


class AppSettings:
    """Singleton quản lý toàn bộ cấu hình ứng dụng Hanoi Metro AI Agent."""
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.vlm = VLMConfig()
        self.llm = LLMConfig()
        self.embedding = EmbeddingConfig()
        self.qdrant = QdrantConfig()
        self.ingestion = IngestionConfig()
        self.chunking = ChunkingConfig()
        self.retrieval = RetrievalConfig()
        self.agent = AgentConfig()

        for attr in ("sqlite_path", "qdrant_local_path", "synonym_dict_path", "blocklist_dir"):
            value = getattr(self.retrieval, attr)
            if value is not None and not value.is_absolute():
                setattr(self.retrieval, attr, self.project_root / value)

        # Chuẩn hóa đường dẫn tương đối theo project root
        if not self.ingestion.raw_pdfs_dir.is_absolute():
            self.ingestion.raw_pdfs_dir = self.project_root / self.ingestion.raw_pdfs_dir
        if not self.ingestion.processed_md_dir.is_absolute():
            self.ingestion.processed_md_dir = self.project_root / self.ingestion.processed_md_dir
        if not self.ingestion.state_db_path.is_absolute():
            self.ingestion.state_db_path = self.project_root / self.ingestion.state_db_path

        # Tự động tạo thư mục làm việc nếu chưa tồn tại
        self._ensure_directories()
        self._setup_logging()

    def _ensure_directories(self) -> None:
        """Đảm bảo các thư mục dữ liệu cần thiết được khởi tạo."""
        self.ingestion.raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.ingestion.processed_md_dir.mkdir(parents=True, exist_ok=True)
        self.ingestion.state_db_path.parent.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """Khởi tạo hệ thống logging đồng nhất cho toàn dự án."""
        level = getattr(logging, self.ingestion.log_level.upper(), logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

        root_logger = logging.getLogger("hanoi_metro")
        root_logger.setLevel(level)
        if not root_logger.handlers:
            root_logger.addHandler(handler)

    def get_vlm_client(self):
        """Khởi tạo OpenAI client cho VLM (tương thích OpenAI, vLLM, Ollama)."""
        from openai import OpenAI
        return OpenAI(
            base_url=self.vlm.base_url,
            api_key=self.vlm.api_key,
            timeout=float(self.vlm.timeout_seconds),
        )

    def get_llm_client(self):
        """Khởi tạo OpenAI client cho LLM."""
        from openai import OpenAI
        return OpenAI(
            base_url=self.llm.base_url,
            api_key=self.llm.api_key,
            timeout=float(self.llm.timeout_seconds),
        )

    def get_embedding_client(self):
        """Khởi tạo OpenAI client cho Embedding Model."""
        from openai import OpenAI
        return OpenAI(
            base_url=self.embedding.base_url,
            api_key=self.embedding.api_key,
            timeout=float(self.embedding.timeout_seconds),
        )

    def get_qdrant_client(self):
        """Khởi tạo Qdrant Client tương tác với Vector DB."""
        from qdrant_client import QdrantClient
        return QdrantClient(
            host=self.qdrant.host,
            port=self.qdrant.port,
            api_key=self.qdrant.api_key,
            timeout=10.0,
        )


# Export singleton instance
settings = AppSettings()
