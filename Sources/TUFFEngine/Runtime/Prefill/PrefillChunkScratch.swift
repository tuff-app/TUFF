import Metal

package typealias PrefillChunkScratchLayout = TUFFRuntimeDescriptions.PrefillChunkScratchLayout

struct PrefillChunkScratchBuffers {
    let layout: PrefillChunkScratchLayout
    let hidden: MTLBuffer
    let normed: MTLBuffer
    let q: MTLBuffer
    let kStage: MTLBuffer
    let vStage: MTLBuffer
    let attentionOutput: MTLBuffer
    let denseX: MTLBuffer
    let routedX: MTLBuffer
    let routerX: MTLBuffer
    let h1: MTLBuffer
    let h2: MTLBuffer
    let routePartials: MTLBuffer
    let routeIDs: MTLBuffer
    let routeWeights: MTLBuffer
    let sharedGateScratch: MTLBuffer
    let sharedUpScratch: MTLBuffer
    let sharedActScratch: MTLBuffer
    let routedGateUpActScratch: MTLBuffer
    let routedDownScratch: MTLBuffer
    // Qwen 3.6 additions. Placeholder-sized (1 element) when the arch does
    // not use them, so the struct stays non-optional.
    let attnQ: MTLBuffer
    let attnGate: MTLBuffer
    let gdnConvOut: MTLBuffer
    let gdnZ: MTLBuffer
    let gdnA: MTLBuffer
    let gdnB: MTLBuffer
    let gdnY: MTLBuffer
    let sharedScalarGate: MTLBuffer
    let pleIdentity: MTLBuffer
    let pleContext: MTLBuffer
    let pleLayer: MTLBuffer
    let pleGate: MTLBuffer
    // Qwen4-Exp hyper-connections. Placeholder-sized when streamCount == 1.
    let hcMixed: MTLBuffer
    let hcLowRank: MTLBuffer
    let hcMixUp: MTLBuffer
    let hcInject: MTLBuffer
    let ngramEmbedding: MTLBuffer

    static func allocate(device: MTLDevice,
                         layout: PrefillChunkScratchLayout) throws -> PrefillChunkScratchBuffers {
        func privateBuffer(_ elements: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * MemoryLayout<Float16>.stride,
                options: .storageModePrivate)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        func sharedBuffer(_ bytes: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(bytes, 1),
                                                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        return PrefillChunkScratchBuffers(
            layout: layout,
            hidden: try privateBuffer(layout.hiddenElements, label: "prefill.hidden"),
            normed: try privateBuffer(layout.normedElements, label: "prefill.normed"),
            q: try privateBuffer(layout.qElements, label: "prefill.q"),
            kStage: try privateBuffer(layout.kStageElements, label: "prefill.kStage"),
            vStage: try privateBuffer(layout.vStageElements, label: "prefill.vStage"),
            attentionOutput: try privateBuffer(layout.attentionOutputElements, label: "prefill.attnOut"),
            denseX: try privateBuffer(layout.denseXElements, label: "prefill.denseX"),
            routedX: try privateBuffer(layout.routedXElements, label: "prefill.routedX"),
            routerX: try privateBuffer(layout.routerXElements, label: "prefill.routerX"),
            h1: try privateBuffer(layout.h1Elements, label: "prefill.h1"),
            h2: try privateBuffer(layout.h2Elements, label: "prefill.h2"),
            routePartials: try privateBuffer(layout.routePartialElements, label: "prefill.routePartials"),
            routeIDs: try sharedBuffer(layout.routeIDElements * MemoryLayout<UInt32>.stride,
                                       label: "prefill.routeIDs"),
            routeWeights: try sharedBuffer(layout.routeWeightElements * MemoryLayout<Float16>.stride,
                                           label: "prefill.routeWeights"),
            sharedGateScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                                 label: "prefill.sharedGateScratch"),
            sharedUpScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                               label: "prefill.sharedUpScratch"),
            sharedActScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                                label: "prefill.sharedActScratch"),
            routedGateUpActScratch: try privateBuffer(layout.routedGateUpActElements,
                                                      label: "prefill.routedGateUpActScratch"),
            routedDownScratch: try privateBuffer(layout.routedDownOutputElements,
                                                 label: "prefill.routedDownScratch"),
            attnQ: try privateBuffer(layout.attnQElements, label: "prefill.attnQ"),
            attnGate: try privateBuffer(layout.attnGateElements, label: "prefill.attnGate"),
            gdnConvOut: try privateBuffer(layout.gdnConvOutElements, label: "prefill.gdnConvOut"),
            gdnZ: try privateBuffer(layout.gdnZElements, label: "prefill.gdnZ"),
            gdnA: try privateBuffer(layout.gdnAElements * layout.gdnScalarElementBytes / MemoryLayout<Float16>.stride,
                                    label: "prefill.gdnA"),
            gdnB: try privateBuffer(layout.gdnBElements * layout.gdnScalarElementBytes / MemoryLayout<Float16>.stride,
                                    label: "prefill.gdnB"),
            gdnY: try privateBuffer(layout.gdnYElements, label: "prefill.gdnY"),
            sharedScalarGate: try privateBuffer(layout.sharedScalarGateBufferElements,
                                                label: "prefill.sharedScalarGate"),
            pleIdentity: try privateBuffer(layout.pleIdentityElements,
                                           label: "prefill.pleIdentity"),
            pleContext: try privateBuffer(layout.pleContextElements,
                                          label: "prefill.pleContext"),
            pleLayer: try privateBuffer(layout.pleLayerElements,
                                        label: "prefill.pleLayer"),
            pleGate: try privateBuffer(layout.pleGateElements,
                                       label: "prefill.pleGate"),
            hcMixed: try privateBuffer(layout.hcMixedElements,
                                       label: "prefill.hcMixed"),
            hcLowRank: try privateBuffer(layout.hcLowRankElements,
                                         label: "prefill.hcLowRank"),
            hcMixUp: try privateBuffer(layout.hcMixUpElements,
                                       label: "prefill.hcMixUp"),
            hcInject: try privateBuffer(layout.hcInjectElements,
                                        label: "prefill.hcInject"),
            ngramEmbedding: try sharedBuffer(
                max(layout.ngramEmbeddingElements, 1) * MemoryLayout<Float16>.stride,
                label: "prefill.ngramEmbedding"))
    }
}
