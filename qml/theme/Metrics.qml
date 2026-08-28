import QtQuick

QtObject {
    readonly property int referenceWidth: 1920
    readonly property int referenceHeight: 1080
    readonly property int minimumWindowWidth: 1280
    readonly property int minimumWindowHeight: 720

    readonly property int radiusSmall: 4
    readonly property int radiusMedium: 8
    readonly property real radiusLarge: 20.99
    readonly property real controlRadiusLarge: 29.44

    readonly property int spacingTiny: 6
    readonly property int spacingSmall: 12
    readonly property int spacingMedium: 20
    readonly property int spacingLarge: 32

    readonly property int iconButtonSize: 36
    readonly property int iconSizeSmall: 16
    readonly property int iconSizeMedium: 20
    readonly property int iconSizeLarge: 24

    readonly property real photoCardReferenceWidth: 252.51
    readonly property real photoCardImageWidth: 252.51
    readonly property real photoCardImageHeight: 168.34
    readonly property real photoCardAspectRatio: photoCardImageWidth / photoCardImageHeight
    readonly property real photoCardRadius: 10
    readonly property real photoCardFooterHeight: 38.45
    readonly property real photoCardReferenceHeight: 206.79
    readonly property real photoCardHorizontalGap: 17

    readonly property real projectCardReferenceWidth: photoCardReferenceWidth
    readonly property real projectCardReferenceHeight: photoCardReferenceHeight

    readonly property real toolbarDropdownWidth: 177.59
    readonly property real toolbarExportButtonWidth: 204.46
    readonly property real toolbarButtonHeight: 41.99
    readonly property real toolbarButtonRadius: 20.99
    readonly property real searchFieldWidth: 320

    readonly property real sidebarPreferredWidth: 430
    readonly property real sidebarNavRowWidth: 274.71
    readonly property real sidebarNavRowHeight: 35.15
    readonly property real sidebarNavRowRadius: 17.57
    readonly property real sidebarCheckboxSize: 12.56
    readonly property real sidebarLastScanWidth: 274.71
    readonly property real sidebarLastScanHeight: 94.55
    readonly property real sidebarLastScanRadius: 21.35

    readonly property real inspectorPreferredWidth: 340
    readonly property real scoreBarWidth: 287.60
    readonly property real scoreBarHeight: 27.07
    readonly property real scoreBarRadius: 13.53
    readonly property real personAvatarSize: 70.47
    readonly property real personAvatarRadius: 24.53
    readonly property real notesFieldWidth: 287.60
    readonly property real notesFieldHeight: 51.84
    readonly property real notesFieldRadius: 21.35

    readonly property real authFieldWidth: 445.98
    readonly property real authFieldHeight: 58.88
    readonly property real authFieldRadius: 29.44
    readonly property real authPrimaryButtonWidth: 445.98
    readonly property real authPrimaryButtonHeight: 58.88
    readonly property real authPrimaryButtonRadius: 29.44
    readonly property real authSocialButtonSize: 67.66
    readonly property real authSocialButtonRadius: 24.53
}
