"""
Hanoi Metro AI Agent - User-safe Error Messages.
Ánh xạ lỗi ở ranh giới giao tiếp (MCP, dịch vụ hỏi đáp) thành thông báo tiếng Việt đã kiểm soát: không lộ exception,
địa chỉ máy chủ, khóa API hay nội dung truy vấn. Chi tiết kỹ thuật chỉ ghi vào log máy chủ.
"""

import logging

from src.agent.assistant import DocumentNotFound
from src.agent.llm import LLMError
from src.indexing.indexer import BlockedDocumentError, BlocklistNotConfigured

logger = logging.getLogger("hanoi_metro.errors")

_LLM_MESSAGES = {
    "llm_unavailable": "Mô hình ngôn ngữ đang không phản hồi. Vui lòng thử lại sau hoặc báo quản trị hệ thống.",
    "llm_timeout": "Mô hình ngôn ngữ xử lý quá lâu. Hãy thu hẹp câu hỏi hoặc thử lại.",
    "llm_rate_limited": "Đã vượt hạn mức gọi API của mô hình ngôn ngữ. Vui lòng thử lại sau ít phút; nếu lặp lại, báo quản trị hệ thống nâng hạn mức API.",
}


def user_message(error: Exception) -> str:
    if isinstance(error, DocumentNotFound):
        return f"Không tìm thấy văn bản: {error}"
    if isinstance(error, BlocklistNotConfigured):
        logger.error(f"Từ chối gửi dữ liệu ra ngoài vì thiếu danh sách chặn: {error}")
        return ("Hệ thống chưa cấu hình danh sách chặn tài liệu mật nên tạm dừng gửi dữ liệu tới dịch vụ bên ngoài. "
                "Vui lòng báo quản trị hệ thống kiểm tra BLOCKLIST_DIR.")
    if isinstance(error, BlockedDocumentError):
        logger.error(f"Đã chặn gửi tài liệu mật ra ngoài: {error}")
        return "Yêu cầu liên quan đến tài liệu có dấu chỉ độ mật nên không được xử lý bằng dịch vụ bên ngoài."
    if isinstance(error, LLMError):
        logger.error(f"LLM lỗi ({error.code}): {error}")
        return _LLM_MESSAGES.get(error.code, _LLM_MESSAGES["llm_unavailable"])
    if isinstance(error, ValueError):
        return f"Tham số không hợp lệ: {error}"
    logger.exception("Lỗi không mong đợi", exc_info=error)
    return "Hệ thống tra cứu gặp lỗi nội bộ. Vui lòng thử lại hoặc báo quản trị hệ thống."
