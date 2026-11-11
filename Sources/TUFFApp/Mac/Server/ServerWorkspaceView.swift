import AppKit
import SwiftUI
import TUFFAppCore
import TUFFAppServer
import TUFFModelCatalog
import TUFFMacPresentation

/// The Background API: a login item that serves every installed model on a
/// loopback port, or to the local network with an API key, whether or not
/// the app is open.
struct ServerWorkspaceView: View {
    let model: AppModel
    @Bindable var controller: AppBackgroundAPIController
    @State private var portText = ""
    @State private var copiedURL = false
    @State private var copiedKey = false
    @State private var showsKey = false
    @FocusState private var portIsFocused: Bool

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                WorkspaceTitle(
                    title: "Server")
                if controller.isAvailable {
                    statusCard
                    configurationCard
                    activityCard
                } else {
                    cloneBuildCard
                }
            }
            .workspaceColumn()
        }
        .workspacePageChrome()
        .task {
            guard controller.isAvailable else { return }
            portText = String(controller.settings.port)
            while !Task.isCancelled {
                await controller.refreshStatus()
                do { try await Task.sleep(for: .seconds(2)) } catch { return }
            }
        }
    }

    // MARK: - Cards

    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 12) {
                ServerIcon(isActive: controller.status != nil)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Background API").appFont(.headline)
                    Text(statusText)
                        .appFont(.callout)
                        .foregroundStyle(TUFFDesign.secondaryTextColor)
                        .contentTransition(.opacity)
                }
                Spacer()
                if controller.isChanging {
                    ProgressView().controlSize(.small)
                        .transition(.opacity)
                }
                Toggle("Background API", isOn: Binding(
                    get: { controller.settings.enabled },
                    set: { value in controller.update { $0.enabled = value } }))
                    .toggleStyle(TUFFSwitchToggleStyle())
                    .labelsHidden()
                    .disabled(controller.isChanging)
                    .accessibilityHint("Starts or stops the login item that serves installed models")
            }
            Text("Runs even when TUFF is closed.")
                .appFont(.callout)
                .foregroundStyle(TUFFDesign.secondaryTextColor)
                .help("Each request names a model, which loads on demand. One model runs at a time.")
            if controller.requiresApproval {
                HStack {
                    Label("Allow TUFF in Login Items to start the listener.",
                          systemImage: "exclamationmark.triangle.fill")
                        .appFont(.callout)
                    Spacer()
                    Button("Open Login Items", action: controller.openLoginItems)
                }
                .transition(.opacity)
            }
            if let message = controller.message {
                Text(message)
                    .appFont(.caption)
                    .foregroundStyle(TUFFDesign.secondaryTextColor)
                    .textSelection(.enabled)
                    .transition(.opacity)
            }
            Divider()
            HStack(spacing: 12) {
                Text(controller.endpoint)
                    .appFont(.body.monospaced())
                    .textSelection(.enabled)
                Spacer()
                Button {
                    copy(controller.endpoint)
                    copiedURL = true
                } label: {
                    Label(copiedURL ? "Copied" : "Copy URL",
                          systemImage: copiedURL ? "checkmark" : "doc.on.doc")
                        .contentTransition(.symbolEffect(.replace))
                }
                .accessibilityHint("Copies the loopback endpoint")
                .task(id: copiedURL) {
                    guard copiedURL else { return }
                    try? await Task.sleep(for: .seconds(1.5))
                    copiedURL = false
                }
            }
        }
        // Turning the server on or off: the status line, the spinner and
        // any message change in place rather than snapping.
        .animation(TUFFMotion.standard, value: statusText)
        .animation(TUFFMotion.standard, value: controller.isChanging)
        .animation(TUFFMotion.standard, value: controller.requiresApproval)
        .animation(TUFFMotion.standard, value: controller.message)
        .tuffCard()
    }

    /// Label on the left, control on the right, one row each with a divider
    /// between, as in Settings.
    private var configurationCard: some View {
        let appleReady = AppleModelAvailability.current().canGenerate
        let appleID = AppleModelKind.onDevice.rawValue
        let availableIDs = installedModels.map(\.apiModelID) + (appleReady ? [appleID] : [])
        return VStack(alignment: .leading, spacing: 0) {
            Text("Configuration")
                .appFont(.headline)
                .padding(.bottom, TUFFDesign.Space.xxs)
            if availableIDs.isEmpty {
                Text("Install a model in the Models tab to serve it.")
                    .appFont(.callout)
                    .foregroundStyle(TUFFDesign.secondaryTextColor)
                    .configurationRow()
            } else {
                LabeledContent("Default model") {
                    GlassMenuPicker("Default model", selection: Binding(
                        get: { controller.defaultModelID },
                        set: { value in controller.update { $0.defaultModel = value } }),
                        options: availableIDs + (availableIDs.contains(controller.defaultModelID)
                            ? [] : [controller.defaultModelID]),
                        label: { id in
                            if id == appleID {
                                return AppleModelKind.onDevice.name + (appleReady ? "" : " (unavailable)")
                            }
                            if let installed = installedModels.first(where: { $0.apiModelID == id }) {
                                return installed.displayName
                            }
                            // By its name where the catalog knows it, not
                            // the identifier requests use ("gemma4").
                            let name = model.installs.map(\.descriptor)
                                .first { $0.apiModelID == id }?.displayName
                                ?? controller.settings.defaultModel
                            return "\(name) (not installed)"
                        })
                    .labelsHidden()
                    .fixedSize()
                    .accessibilityHint("The model used by requests that name default")
                }
                .help("Used when a request names \"default\". Separate from the model chosen in Chat.")
                .configurationRow()
            }
            Divider()
            LabeledContent("Unload model") {
                GlassMenuPicker("Unload model", selection: Binding(
                    get: { controller.settings.unloadDelay },
                    set: { value in controller.update { $0.unloadDelay = value } }),
                    options: TUFFModelUnloadDelay.choices,
                    label: { $0.label })
                .labelsHidden()
                .fixedSize()
            }
            .configurationRow()
            Divider()
            LabeledContent("Port") {
                TextField("Port", text: $portText)
                    .labelsHidden()
                    .tuffField()
                    .multilineTextAlignment(.trailing)
                    .frame(width: 80)
                    .focused($portIsFocused)
                    .onSubmit {
                        controller.updatePort(portText)
                        portIsFocused = false
                    }
                    .onChange(of: portIsFocused) { _, focused in
                        if !focused { controller.updatePort(portText) }
                    }
                    .onChange(of: controller.settings.port) { _, port in
                        if !portIsFocused { portText = String(port) }
                    }
            }
            .configurationRow()
            Divider()
            Toggle("Allow other devices on your network", isOn: Binding(
                get: { controller.settings.listensBeyondLoopback },
                set: { value in controller.setNetworkAccess(value) }))
                .toggleStyle(TUFFRowToggleStyle())
                .accessibilityHint("Listens on every network interface and requires the API key from other devices")
                .configurationRow()
            if controller.settings.listensBeyondLoopback {
                networkAccess
                    .padding(.bottom, TUFFDesign.Space.xs)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
            Text(controller.settings.listensBeyondLoopback
                 ? "Other devices must send the API key as Authorization: Bearer <key> or x-api-key. Requests from this Mac need no key. Anyone with the key can use your models, so only allow this on a network you trust. Chat and the API share one memory budget, so the API declines a request while Chat holds a model the second would not fit beside."
                 : "The server listens only on 127.0.0.1, so only this Mac can reach it, and it needs no key. Chat and the API share one memory budget, so the API declines a request while Chat holds a model the second would not fit beside.")
                .appFont(.caption)
                .foregroundStyle(TUFFDesign.secondaryTextColor)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, TUFFDesign.Space.xxs)
        }
        .labeledContentStyle(TUFFRowLabeledContentStyle())
        .disabled(controller.isChanging)
        .animation(TUFFMotion.standard, value: controller.settings.listensBeyondLoopback)
        .tuffCard()
    }

    private var networkAccess: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("From other devices")
                    .appFont(.caption.weight(.semibold))
                    .foregroundStyle(TUFFDesign.secondaryTextColor)
                let endpoints = controller.networkEndpoints
                if endpoints.isEmpty {
                    Text("This Mac has no network address right now.")
                        .appFont(.callout)
                        .foregroundStyle(TUFFDesign.secondaryTextColor)
                }
                ForEach(endpoints, id: \.self) { endpoint in
                    Text(endpoint)
                        .appFont(.body.monospaced())
                        .textSelection(.enabled)
                }
            }
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("API key")
                        .appFont(.caption.weight(.semibold))
                        .foregroundStyle(TUFFDesign.secondaryTextColor)
                    if let key = controller.apiKey {
                        Text(showsKey ? key : String(repeating: "•", count: 24))
                            .appFont(.body.monospaced())
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                            .accessibilityLabel(showsKey ? key : "Hidden")
                    } else {
                        Text("Created when the server starts")
                            .appFont(.callout)
                            .foregroundStyle(TUFFDesign.secondaryTextColor)
                    }
                }
                Spacer()
                if let key = controller.apiKey {
                    Button(showsKey ? "Hide" : "Show") { showsKey.toggle() }
                    Button {
                        copy(key)
                        copiedKey = true
                    } label: {
                        Label(copiedKey ? "Copied" : "Copy Key",
                              systemImage: copiedKey ? "checkmark" : "key")
                            .contentTransition(.symbolEffect(.replace))
                    }
                    .accessibilityHint("Copies the API key other devices send")
                    .task(id: copiedKey) {
                        guard copiedKey else { return }
                        try? await Task.sleep(for: .seconds(1.5))
                        copiedKey = false
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .tuffInset()
    }

    private var activityCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Activity").appFont(.headline)
                Spacer()
                Button {
                    controller.openLog()
                } label: {
                    Label("Open Log", systemImage: "doc.text.magnifyingglass")
                }
                .disabled(!controller.hasLog)
                .accessibilityHint("Opens the Background API log in Console")
            }
            HStack(spacing: 12) {
                ServerMetric(
                    title: loadedModelCaption,
                    value: loadedModelText,
                    systemImage: "cube",
                    accent: controller.status?.residentModel != nil)
                ServerMetric(
                    title: "Active",
                    value: controller.status.map { "\($0.activeRequests)" } ?? "–",
                    systemImage: "bolt.horizontal.circle",
                    accent: (controller.status?.activeRequests ?? 0) > 0)
                ServerMetric(
                    title: "Queued",
                    value: controller.status.map { "\($0.queuedRequests)" } ?? "–",
                    systemImage: "list.number",
                    accent: (controller.status?.queuedRequests ?? 0) > 0)
            }
        }
        .tuffCard()
    }

    /// Clone builds have no app bundle to register a login item from, so they
    /// run the same router in Terminal.
    private var cloneBuildCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 12) {
                ServerIcon(isActive: false)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Background API").appFont(.headline)
                    Text("Available in the packaged TUFF app")
                        .appFont(.callout)
                        .foregroundStyle(TUFFDesign.secondaryTextColor)
                }
            }
            Text("This build cannot register a login item. From the repository, run the same server in Terminal. It serves every model in scratch/ until you press Control-C.")
                .appFont(.callout)
                .foregroundStyle(TUFFDesign.secondaryTextColor)
            Divider()
            HStack(spacing: 12) {
                Text(Self.cloneCommand)
                    .appFont(.body.monospaced())
                    .textSelection(.enabled)
                Spacer()
                Button {
                    copy(Self.cloneCommand)
                } label: {
                    Label("Copy Command", systemImage: "doc.on.doc")
                }
            }
        }
        .tuffCard()
    }

    static let cloneCommand = "swift run -c release TUFFServer --models-root scratch"

    // MARK: - Derived values

    private var installedModels: [AppModelInstallDescriptor] {
        model.installs.filter(\.isInstalled).map(\.descriptor)
    }

    private var statusText: String {
        let settings = controller.settings
        if !settings.enabled { return "Off" }
        if controller.requiresApproval { return "Waiting for approval in Login Items" }
        if controller.status != nil {
            return settings.listensBeyondLoopback
                ? "Listening on port \(settings.port) for this Mac and your network"
                : "Listening on 127.0.0.1:\(settings.port)"
        }
        return "Starting the listener…"
    }

    private var loadedModelText: String {
        guard let status = controller.status else { return "–" }
        if let transition = status.transition { return transition }
        guard let id = status.residentModel else { return "None" }
        return installedModels.first { $0.apiModelID == id }?.displayName ?? id
    }

    private var loadedModelCaption: String {
        guard let seconds = controller.status?.idleUnloadInSeconds else { return "Loaded model" }
        return "Loaded model · unloads in \(Duration.seconds(seconds).formatted(.units(allowed: [.minutes, .seconds], width: .narrow)))"
    }

    private func copy(_ string: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(string, forType: .string)
    }
}

private struct ServerIcon: View {
    let isActive: Bool

    var body: some View {
        Image(systemName: "network")
            .appFont(.title2)
            .foregroundStyle(isActive ? TUFFMacTheme.accentTextColor : TUFFDesign.secondaryTextColor)
            .frame(width: 32)
            .accessibilityHidden(true)
    }
}

private struct ServerMetric: View {
    let title: String
    let value: String
    let systemImage: String
    let accent: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: systemImage)
                .foregroundStyle(accent ? TUFFMacTheme.accentTextColor : .secondary)
            Text(value)
                .appFont(.title3.weight(.semibold).monospacedDigit())
                .lineLimit(1)
                .truncationMode(.middle)
            Text(title)
                .appFont(.caption)
                .foregroundStyle(TUFFDesign.secondaryTextColor)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .tuffInset()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(title)
        .accessibilityValue(value)
    }
}

private extension View {
    /// One row of a settings-style card: the height and spacing of a
    /// Settings row, so the two screens read the same.
    func configurationRow() -> some View {
        frame(maxWidth: .infinity, minHeight: 28, alignment: .leading)
            .padding(.vertical, 9)
    }
}
