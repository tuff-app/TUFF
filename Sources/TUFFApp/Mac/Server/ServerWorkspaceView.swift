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
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @State private var portText = ""
    @State private var copiedURL = false
    @State private var copiedKey = false
    @State private var showsKey = false
    @FocusState private var portIsFocused: Bool

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                WorkspaceTitle(
                    title: "Server",
                    subtitle: "Use installed models through an OpenAI-compatible local endpoint.")
                if controller.isAvailable {
                    statusCard
                    configurationCard
                    activityCard
                } else {
                    cloneBuildCard
                }
            }
            .frame(maxWidth: 920, alignment: .leading)
            .padding(28)
            // Centred in a wide window rather than pinned to the sidebar.
            .frame(maxWidth: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Color(nsColor: .windowBackgroundColor))
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
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if controller.isChanging {
                    ProgressView().controlSize(.small)
                }
                Toggle("Background API", isOn: Binding(
                    get: { controller.settings.enabled },
                    set: { value in controller.update { $0.enabled = value } }))
                    .toggleStyle(.switch)
                    .labelsHidden()
                    .disabled(controller.isChanging)
                    .accessibilityHint("Starts or stops the login item that serves installed models")
            }
            Text("Serves installed models while TUFF is open or closed. Each request names a model, which loads on demand; one model runs at a time.")
                .appFont(.callout)
                .foregroundStyle(.secondary)
            if controller.requiresApproval {
                HStack {
                    Label("Allow TUFF in Login Items to start the listener.",
                          systemImage: "exclamationmark.triangle")
                        .appFont(.callout)
                    Spacer()
                    Button("Open Login Items", action: controller.openLoginItems)
                }
            }
            if let message = controller.message {
                Text(message)
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            Divider()
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Endpoint")
                        .appFont(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Text(controller.endpoint)
                        .appFont(.body.monospaced())
                        .textSelection(.enabled)
                }
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
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(cardBackground, in: RoundedRectangle(cornerRadius: 16))
    }

    private var configurationCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Configuration").appFont(.headline)
            if installedModels.isEmpty {
                Text("Install a model on the Models screen to serve it.")
                    .appFont(.callout)
                    .foregroundStyle(.secondary)
            } else {
                Picker("Default model", selection: Binding(
                    get: { controller.defaultModelID },
                    set: { value in controller.update { $0.defaultModel = value } })) {
                    ForEach(installedModels, id: \.apiModelID) { descriptor in
                        Text(descriptor.displayName).tag(descriptor.apiModelID)
                    }
                    if !installedModels.contains(where: { $0.apiModelID == controller.defaultModelID }) {
                        Text("\(controller.settings.defaultModel) (not installed)")
                            .tag(controller.defaultModelID)
                    }
                }
                .accessibilityHint("The model used by requests that name default")
                .help("Used when a request names \"default\". Separate from the model chosen in Chat.")
            }
            Picker("Unload model", selection: Binding(
                get: { controller.settings.unloadDelay },
                set: { value in controller.update { $0.unloadDelay = value } })) {
                ForEach(TUFFModelUnloadDelay.choices, id: \.self) { delay in
                    Text(delay.label).tag(delay)
                }
            }
            LabeledContent("Port") {
                TextField("Port", text: $portText)
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 120)
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
            .fixedSize()
            Toggle("Allow other devices on your network", isOn: Binding(
                get: { controller.settings.listensBeyondLoopback },
                set: { value in controller.setNetworkAccess(value) }))
                .accessibilityHint("Listens on every network interface and requires the API key from other devices")
            if controller.settings.listensBeyondLoopback {
                networkAccess
            }
            Text(controller.settings.listensBeyondLoopback
                 ? "Other devices must send the API key as Authorization: Bearer <key> or x-api-key. Requests from this Mac need no key. Anyone with the key can use your models, so only allow this on a network you trust. Chat and the API share one memory budget, so the API declines a request while Chat holds a model the second would not fit beside."
                 : "The server listens only on 127.0.0.1, so only this Mac can reach it, and it needs no key. Chat and the API share one memory budget, so the API declines a request while Chat holds a model the second would not fit beside.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
        }
        .disabled(controller.isChanging)
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(cardBackground, in: RoundedRectangle(cornerRadius: 16))
    }

    private var networkAccess: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("From other devices")
                    .appFont(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                let endpoints = controller.networkEndpoints
                if endpoints.isEmpty {
                    Text("This Mac has no network address right now.")
                        .appFont(.callout)
                        .foregroundStyle(.secondary)
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
                        .foregroundStyle(.secondary)
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
                            .foregroundStyle(.secondary)
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
        .padding(12)
        .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 12))
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
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(cardBackground, in: RoundedRectangle(cornerRadius: 16))
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
                        .foregroundStyle(.secondary)
                }
            }
            Text("This build cannot register a login item. From the repository, run the same server in Terminal. It serves every model in scratch/ until you press Control-C.")
                .appFont(.callout)
                .foregroundStyle(.secondary)
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
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(cardBackground, in: RoundedRectangle(cornerRadius: 16))
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

    private var cardBackground: AnyShapeStyle {
        TUFFMacTheme.surfaceStyle(reduceTransparency: reduceTransparency)
    }

    private func copy(_ string: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(string, forType: .string)
    }
}

private struct ServerIcon: View {
    let isActive: Bool

    var body: some View {
        ZStack {
            Circle().fill(TUFFMacTheme.accentColor.opacity(isActive ? 0.2 : 0.1))
            Image(systemName: "network")
                .appFont(.title2)
                .foregroundStyle(isActive ? TUFFMacTheme.accentColor : .secondary)
        }
        .frame(width: 44, height: 44)
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
                .foregroundStyle(accent ? TUFFMacTheme.accentColor : .secondary)
            Text(value)
                .appFont(.title3.weight(.semibold))
                .lineLimit(1)
                .truncationMode(.middle)
            Text(title)
                .appFont(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(title)
        .accessibilityValue(value)
    }
}
