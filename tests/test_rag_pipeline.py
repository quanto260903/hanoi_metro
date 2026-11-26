"""
Hanoi Metro AI Agent - End-to-End RAG Pipeline Test & Verification Script.
Kiểm thử toàn diện Giai đoạn 2:
1. Tự động đọc và cắt nhỏ tài liệu Markdown (.md) trong data/processed_md/ (Chunking + Contextual Header).
2. Sinh Hybrid Embedding (Dense + Sparse) và Upsert vào Qdrant Vector Database.
3. Nhận câu hỏi tiếng Việt từ terminal hoặc dòng lệnh, thực hiện Hybrid Search (RRF Top 15) và Reranker (Top 5).
4. Tổng hợp câu trả lời chuyên sâu từ LLM (Qwen/Ollama hoặc OpenAI-compatible) kèm trích dẫn nguồn (Citations).
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Thêm thư mục gốc dự án vào PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import settings
from src.indexing.chunker import MarkdownDocumentChunker
from src.indexing.embedder import HybridEmbedder, QdrantIndexer
from src.retrieval.hybrid_search import HybridRetriever, RetrievalResult

logger = logging.getLogger("hanoi_metro.test_pipeline")


def ensure_data_indexed(reindex: bool = False) -> int:
    """Đảm bảo dữ liệu từ data/processed_md/ đã được chunking và indexing vào Qdrant."""
    indexer = QdrantIndexer()
    md_dir = settings.ingestion.processed_md_dir

    if not md_dir.exists():
        print(f"[!] Thư mục {md_dir} không tồn tại. Vui lòng chạy Ingestion Pipeline trước!")
        return 0

    # Kiểm tra xem collection đã có điểm dữ liệu nào chưa
    collection_ready = False
    try:
        col_info = indexer.client.get_collection(collection_name=indexer.collection_name)
        points_count = col_info.points_count or 0
        if points_count > 0 and not reindex:
            collection_ready = True
            print(f"[+] Collection '{indexer.collection_name}' đã sẵn sàng với {points_count} vector points.")
            return points_count
    except Exception:
        pass

    print(f"\n[*] Đang tiến hành nạp dữ liệu từ thư mục '{md_dir}' vào Qdrant...")
    chunker = MarkdownDocumentChunker(max_chunk_size=1200)
    chunks = chunker.chunk_directory(md_dir)

    if not chunks:
        print("[!] Không tìm thấy file .md nào trong thư mục processed_md.")
        return 0

    print(f"[+] Đã chia tách thành công {len(chunks)} chunks.")
    total_upserted = indexer.upsert_chunks(chunks)
    print(f"[+] Hoàn tất nạp {total_upserted} chunks vào Qdrant Vector DB.\n")
    return total_upserted


def generate_llm_answer(query: str, retrieved_chunks: List[RetrievalResult]) -> str:
    """
    Gửi ngữ cảnh Top Chunks và câu hỏi của người dùng tới LLM để sinh câu trả lời
    chính xác, kèm trích dẫn số liệu và tài liệu nguồn.
    """
    if not retrieved_chunks:
        return "Hệ thống không tìm thấy tài liệu phù hợp trong cơ sở dữ liệu để trả lời câu hỏi."

    # Xây dựng ngữ cảnh context kèm thông tin nguồn
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"--- TRÍCH DẪN TÀI LIỆU [{i}] ---\n"
            f"Tên file nguồn: {chunk.source_file}\n"
            f"Loại văn bản: {chunk.doc_type} | Trang: {chunk.page_number} | Mục: {chunk.heading_path}\n"
            f"Nội dung:\n{chunk.raw_content}\n"
        )
    context_str = "\n".join(context_blocks)

    system_prompt = (
        "Bạn là Trợ lý AI Chuyên gia Phân tích Tài liệu Kỹ thuật và Nghiệp vụ của Công ty TNHH MTV Đường sắt Hà Nội (Hanoi Metro).\n"
        "Nhiệm vụ của bạn là trả lời các câu hỏi dựa CHÍNH XÁC vào ngữ cảnh tài liệu nội bộ được cung cấp bên dưới.\n\n"
        "Quy tắc trả lời:\n"
        "1. Trả lời bằng tiếng Việt chuẩn xác, súc tích, chuyên nghiệp và có cấu trúc rõ ràng.\n"
        "2. Bảo toàn nguyên vẹn tên thiết bị, số serial, số lượng, tình trạng kỹ thuật, thời gian và địa điểm từ các bảng biểu.\n"
        "3. Ở cuối câu trả lời, LUÔN LUÔN ghi rõ phần 'Nguồn trích dẫn' liệt kê tên file tài liệu, số trang hoặc mục liên quan.\n"
        "4. Nếu trong ngữ cảnh tài liệu KHÔNG có thông tin để trả lời, hãy nói rõ: 'Dựa trên tài liệu hiện có, không tìm thấy thông tin về vấn đề này', tuyệt đối không bịa đặt."
    )

    user_prompt = (
        f"Ngữ cảnh tài liệu nội bộ Hanoi Metro:\n"
        f"{context_str}\n\n"
        f"Câu hỏi của cán bộ / người dùng:\n"
        f"{query}\n\n"
        f"Hãy trả lời chi tiết và kèm trích dẫn nguồn:"
    )

    client = settings.get_llm_client()
    try:
        response = client.chat.completions.create(
            model=settings.llm.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=float(settings.llm.timeout_seconds),
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Không thể kết nối LLM ({e}). Tổng hợp câu trả lời trực tiếp từ Top Chunks.")
        fallback_ans = (
            f"[THÔNG BÁO: Chưa kết nối được tới LLM ({e}), hiển thị trích xuất trực tiếp từ các đoạn văn bản khớp nhất]:\n\n"
        )
        for i, chunk in enumerate(retrieved_chunks[:2], start=1):
            fallback_ans += f"📌 **Nguồn [{i}] ({chunk.source_file} - Mục: {chunk.heading_path}):**\n"
            fallback_ans += f"{chunk.raw_content}\n\n"
        return fallback_ans


def print_search_results(results: List[RetrievalResult]) -> None:
    """In danh sách các chunks được truy xuất đẹp mắt ra màn hình terminal."""
    print("\n" + "=" * 80)
    print(f"📊 KẾT QUẢ TRUY XUẤT HYBRID SEARCH & RERANKER (TOP {len(results)} CHUNKS):")
    print("=" * 80)

    for idx, item in enumerate(results, start=1):
        table_badge = " [BẢNG BIỂU]" if item.has_table else ""
        print(f"[{idx}] Điểm Rerank: {item.score:.2f} | RRF Score: {item.rrf_score:.4f}{table_badge}")
        print(f"    - Nguồn: {item.source_file} (Loại: {item.doc_type})")
        print(f"    - Vị trí: Trang {item.page_number} | Mục: {item.heading_path or 'Toàn văn'}")
        preview = item.raw_content.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:160] + "..."
        print(f"    - Trích lược: {preview}")
        print("-" * 80)


def process_query(retriever: HybridRetriever, query: str, top_k: int = 5) -> None:
    """Xử lý một câu truy vấn hoàn chỉnh từ retrieval đến LLM generation."""
    print(f"\n🔍 Đang xử lý câu hỏi: '{query}'")
    retrieved = retriever.retrieve(query=query, top_k=top_k, top_k_hybrid=15)

    if not retrieved:
        print("[!] Không tìm thấy dữ liệu phù hợp.")
        return

    print_search_results(retrieved)

    print("\n🤖 Đang tổng hợp câu trả lời từ LLM...")
    answer = generate_llm_answer(query, retrieved)

    print("\n" + "=" * 80)
    print("💡 CÂU TRẢ LỜI CỦA HANOI METRO AI AGENT:")
    print("=" * 80)
    print(answer)
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Hanoi Metro AI Agent - Test RAG Pipeline")
    parser.add_argument("--query", "-q", type=str, default="", help="Câu hỏi kiểm thử từ dòng lệnh")
    parser.add_argument("--reindex", action="store_true", help="Buộc nạp lại toàn bộ dữ liệu vào Qdrant")
    parser.add_argument("--top-k", type=int, default=5, help="Số lượng chunk lấy sau khi Rerank")
    args = parser.parse_args()

    print("================================================================================")
    print("       HANOI METRO AI AGENT - KIỂM THỬ PIPELINE RAG (GIAI ĐOẠN 2)               ")
    print("================================================================================")

    # 1. Đảm bảo dữ liệu đã được index
    ensure_data_indexed(reindex=args.reindex)

    # 2. Khởi tạo Hybrid Retriever
    retriever = HybridRetriever()

    # 3. Chạy câu hỏi nếu được truyền qua CLI
    if args.query:
        process_query(retriever, args.query, top_k=args.top_k)
        return

    # 4. Danh sách câu hỏi kiểm thử mẫu dựa trên tài liệu thực tế trong data/processed_md/
    sample_queries = [
        "Thời gian, địa điểm và nhân sự tham gia khảo sát hệ thống AFC diễn ra như thế nào?",
        "Tại Trung tâm dữ liệu DC số 17 Duy Tân, các thiết bị nào bị ghi nhận là bị hỏng?",
        "Thành phần đại diện tham gia kiểm đếm thiết bị tại Trung tâm dữ liệu Duy Tân gồm những ai?",
    ]

    print("[*] Chạy thử nghiệm với 3 câu hỏi mẫu trên dữ liệu thực tế nội bộ Hanoi Metro:")
    for i, q in enumerate(sample_queries, start=1):
        print(f"\n>>> TEST CASE {i} <<<")
        process_query(retriever, q, top_k=args.top_k)

    # 5. Chế độ Terminal Interactive: Người dùng gõ câu hỏi tùy ý
    print("\n" + "#" * 80)
    print(" BẮT ĐẦU CHẾ ĐỘ NHẬP CÂU HỎI TỰ DO TỪ TERMINAL (Gõ 'exit' hoặc 'quit' để thoát)")
    print("#" * 80)

    while True:
        try:
            user_input = input("\n👉 Nhập câu hỏi của bạn: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Tạm biệt! Kết thúc phiên làm việc.")
                break
            process_query(retriever, user_input, top_k=args.top_k)
        except (KeyboardInterrupt, EOFError):
            print("\nĐã dừng tiến trình kiểm thử.")
            break


if __name__ == "__main__":
    main()
