import Foundation
import Darwin

/// Shared building blocks for the resident LM and routed-expert layer writers.
public enum WriterCore {

    /// Tile size for pwrite (and the subsequent SHA-256 hashing pass). Chosen
    /// so per-worker scratch and per-syscall payload both stay well under
    /// the 1 MB BoundedScratch budget.
    public static let tileBytes: Int = 512 * 1024

    /// Copy `size` bytes from `srcShard.base + srcOffset` to file `dstFd` at
    /// `dstOffset`, in pwrite-sized tiles. Pages consumed from the source map
    /// are evicted via madvise after each tile, capping the source-side RSS.
    public static func pwriteTensorRegion(srcShard: MmapHandle,
                                          srcAbsoluteOffset: UInt64,
                                          size: UInt64,
                                          dstFd: Int32, dstPath: String,
                                          dstOffset: UInt64,
                                          audit: RepackAudit) throws {
        var remaining = Int(size)
        var srcOff = srcAbsoluteOffset
        var dstOff = dstOffset
        let tile = WriterCore.tileBytes
        while remaining > 0 {
            try Task.checkCancellation()
            let n = min(remaining, tile)
            let p = srcShard.base.advanced(by: Int(srcOff))
            try Posix.pwriteAll(fd: dstFd, path: dstPath, buf: p, count: n, offset: dstOff)
            audit.recordTile(bytes: n)
            audit.recordWrite(bytes: n)
            audit.recordRead(bytes: n)
            srcShard.adviseDontNeed(offset: srcOff, count: n)
            srcOff += UInt64(n)
            dstOff += UInt64(n)
            remaining -= n
        }
    }

    /// Compute SHA-256 of an entire (presumed-written) file by streaming it
    /// through `tileBytes` pread chunks. Drops pages with `F_NOCACHE` style
    /// behaviour via fcntl. Allocates one bounded scratch buffer.
    public static func hashEntireFile(path: String, size: UInt64,
                                      audit: RepackAudit,
                                      cancellationCheck: () throws -> Void = {}) throws -> String {
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        // Hint the kernel that we will read this file sequentially and then
        // drop it from cache — keeps the post-write working set from blowing
        // up the dev box.
        _ = fcntl(fd, F_NOCACHE, 1)

        let buf = UnsafeMutableRawBufferPointer.allocate(byteCount: WriterCore.tileBytes,
                                                         alignment: 16_384)
        defer { buf.deallocate() }
        if buf.count > audit.largestScratchBytes {
            audit.largestScratchBytes = buf.count
        }

        var hasher = Sha256Stream()
        var off: UInt64 = 0
        let total = Int(size)
        var remaining = total
        while remaining > 0 {
            try cancellationCheck()
            let want = min(remaining, WriterCore.tileBytes)
            let got = pread(fd, buf.baseAddress, want, off_t(off))
            if got <= 0 {
                throw RepackError.preadShort(path: path, expected: want, got: 0, errno: errno)
            }
            hasher.update(UnsafeRawBufferPointer(start: buf.baseAddress, count: got))
            audit.byteCopyTiles &+= 1
            off += UInt64(got)
            remaining -= got
        }
        return hasher.finalizeHexString()
    }

    public static func pwriteFP32AsBF16(srcShard: MmapHandle,
                                        srcAbsoluteOffset: UInt64,
                                        size: UInt64,
                                        dstFd: Int32, dstPath: String,
                                        dstOffset: UInt64,
                                        audit: RepackAudit) throws {
        guard size.isMultiple(of: 4) else {
            throw RepackError.configurationInvalid(
                detail: "FP32 source region has non-element-aligned byte count \(size)")
        }
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: tileBytes, alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)
        var remaining = size
        var source = srcAbsoluteOffset
        var destination = dstOffset
        while remaining > 0 {
            try Task.checkCancellation()
            let count = min(Int(remaining), scratch.count) & ~3
            let sourcePointer = srcShard.base.advanced(by: Int(source))
            memcpy(scratch.baseAddress!, sourcePointer, count)
            let outputCount = convertFP32ToBF16InPlace(scratch.baseAddress!, byteCount: count)
            try Posix.pwriteAll(fd: dstFd, path: dstPath,
                                buf: scratch.baseAddress!, count: outputCount,
                                offset: destination)
            audit.recordTile(bytes: count)
            audit.recordRead(bytes: count)
            audit.recordWrite(bytes: outputCount)
            srcShard.adviseDontNeed(offset: source, count: count)
            remaining -= UInt64(count)
            source += UInt64(count)
            destination += UInt64(outputCount)
        }
    }

    @discardableResult
    public static func convertFP32ToBF16InPlace(_ buffer: UnsafeMutableRawPointer,
                                                byteCount: Int) -> Int {
        precondition(byteCount.isMultiple(of: 4))
        let elementCount = byteCount / 4
        let input = buffer.assumingMemoryBound(to: UInt32.self)
        let output = buffer.assumingMemoryBound(to: UInt16.self)
        for index in 0..<elementCount {
            let bits = UInt32(littleEndian: input[index])
            let exponent = bits & 0x7f80_0000
            let mantissa = bits & 0x007f_ffff
            let bf16: UInt16
            if exponent == 0x7f80_0000, mantissa != 0 {
                // Preserve NaN rather than allowing rounding overflow to turn
                // the largest payload into signed zero.
                bf16 = UInt16(truncatingIfNeeded: bits >> 16) | 0x0040
            } else {
                let rounded = bits &+ 0x7fff &+ ((bits >> 16) & 1)
                bf16 = UInt16(truncatingIfNeeded: rounded >> 16)
            }
            output[index] = bf16.littleEndian
        }
        return elementCount * 2
    }
}
