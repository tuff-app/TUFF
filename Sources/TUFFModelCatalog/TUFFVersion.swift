/// The TUFF release this source tree builds.
///
/// Clone builds have no Info.plist to read a version from, so this constant is
/// what the About panel, `tuff --version` and the server's status route report.
/// `Scripts/package_app.sh` refuses to package a version that disagrees with it,
/// and `Scripts/check_app_version.rb` fails CI when it falls behind the newest
/// published release.
public enum TUFFVersion {
    public static let current = "11.0.1"
}
