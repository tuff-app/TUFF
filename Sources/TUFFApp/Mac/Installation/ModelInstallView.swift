import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// Shown while the selected model is not installed.
///
/// It presents the whole catalog rather than only the selected model, so the
/// first thing a new user does is choose which model to install — and so a
/// second model can be downloaded from here at any time without disturbing the
/// first.
struct ModelInstallView: View {
    let model: AppModel

    var body: some View {
        ScrollView {
            VStack(spacing: 22) {
                identity
                ModelCatalogView(model: model)
                locationCard
            }
            .frame(maxWidth: 620)
            .padding(.horizontal, 28)
            .padding(.vertical, 48)
            .frame(maxWidth: .infinity)
        }
    }

    private var identity: some View {
        VStack(spacing: 10) {
            Image(systemName: "externaldrive.badge.plus")
                .appFont(.system(.largeTitle, design: .rounded))
                .foregroundStyle(TUFFMacTheme.accentTextColor)
                .accessibilityHidden(true)
            Text("Choose a model")
                .appFont(.title.bold())
                .accessibilityHeading(.h1)
            Text("Download a model to start chatting. You can add more later.")
                .appFont(.body)
                .foregroundStyle(TUFFDesign.secondaryTextColor)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var locationCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            DisclosureGroup("Install location") {
                Text(model.modelPathText)
                    .appFont(.caption.monospaced())
                    .foregroundStyle(TUFFDesign.tertiaryTextColor)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                HStack {
                    Spacer()
                    Button("Check Again", action: model.recheckModelAtCurrentLocation)
                        .disabled(model.isInstallingModel)
                        .help("Re-scan every model's directory, "
                              + "for a model installed outside the app")
                }
            }
        }
        .tuffCard()
    }
}
