"""Read Word 97–2003 main-story text via the MS-DOC piece table (no macros)."""
import struct
from pathlib import Path

import olefile


def extract_doc_text(path: Path) -> str:
    try:
        with olefile.OleFileIO(path) as doc:
            word = doc.openstream('WordDocument').read()
            def u16(offset):
                return struct.unpack_from('<H', word, offset)[0]
            def u32(offset):
                return struct.unpack_from('<I', word, offset)[0]
            if u16(0) != 0xA5EC or u16(2) < 0x00C1:
                raise ValueError('仅支持 Word 97–2003 二进制 DOC')
            flags = u16(10)
            if flags & 0x8100:
                raise ValueError('此 Word 已加密，请解密后上传')
            table = doc.openstream('1Table' if flags & 0x200 else '0Table').read()
            offset = 32
            offset += 2 + 2 * u16(offset)
            main_length = u32(offset + 2 + 3 * 4)
            offset += 2 + 4 * u16(offset)
            if u16(offset) < 34:
                raise ValueError('Word 文件头缺少文本索引')
            start = u32(offset + 2 + 33 * 8)
            size = u32(offset + 2 + 33 * 8 + 4)
            clx = table[start:start + size]
            pos = 0
            while pos < len(clx) and clx[pos] == 1:
                pos += 3 + struct.unpack_from('<H', clx, pos + 1)[0]
            if pos >= len(clx) or clx[pos] != 2:
                raise ValueError('Word 文本索引无法解析')
            length = struct.unpack_from('<I', clx, pos + 1)[0]
            plc = clx[pos + 5:pos + 5 + length]
            if len(plc) != length or length < 4 or (length - 4) % 12:
                raise ValueError('Word 文本索引损坏')
            count = (length - 4) // 12
            pieces = []
            for i in range(count):
                cp, end = struct.unpack_from('<II', plc, i * 4)
                # Include text-box and header stories: exam layouts often put all
                # questions in floating text boxes rather than the main story.
                chars = end - cp
                if chars < 0:
                    raise ValueError('Word 字符索引无效')
                fc = struct.unpack_from('<I', plc, 4 * (count + 1) + i * 8 + 2)[0]
                compressed = bool(fc & 0x40000000)
                address = fc & 0x3fffffff
                address = address // 2 if compressed else address
                byte_count = chars if compressed else chars * 2
                raw = word[address:address + byte_count]
                if len(raw) != byte_count:
                    raise ValueError('Word 文本数据不完整')
                pieces.append(raw.decode('cp1252' if compressed else 'utf-16-le', errors='replace'))
            text = ''.join(pieces).replace('\r', '\n').replace('\x07', '\t').replace('\x0b', '\n').replace('\x0c', '\n')
            text = ''.join(c for c in text if c >= ' ' or c in '\n\t')
            if not text.strip():
                raise ValueError('DOC 没有可提取正文，请上传 PDF 以读取扫描页或嵌入对象')
            return text.strip()
    except (OSError, KeyError, struct.error) as exc:
        raise ValueError('不是有效的 Word DOC，或文件已损坏') from exc
