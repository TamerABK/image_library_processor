import QtQuick
import QtQuick.Layouts
import "../theme" as Room36Theme
import "../components" as Components

Item {
    id: root

    signal searchChanged(string text)
    signal subscriptionRequested()
    signal settingsRequested()
    signal profileRequested()

    Room36Theme.Theme { id: theme }
    Room36Theme.Metrics { id: metrics }

    readonly property bool referenceLayout: width >= 1500
    readonly property real margin: referenceLayout ? metrics.homeHorizontalMargin : 48
    readonly property real logoWidth: referenceLayout ? metrics.homeLogoWidth : 210
    readonly property real logoHeight: logoWidth / 3.678

    implicitHeight: referenceLayout ? 156 : 142

    Image {
        id: logo
        x: root.margin
        y: root.referenceLayout ? 66.96 : 32
        width: root.logoWidth
        height: root.logoHeight
        source: theme.brandingUrl("logo_blue.svg")
        cache: true
        fillMode: Image.PreserveAspectFit
    }

    RowLayout {
        id: actions
        x: root.referenceLayout
            ? root.width - implicitWidth - 84
            : Math.max(root.margin, root.width - implicitWidth - root.margin)
        y: root.referenceLayout ? 72.82 : 58
        spacing: 28

        Components.SearchField {
            id: search
            Layout.preferredWidth: root.referenceLayout ? metrics.homeSearchFieldWidth : 360
            Layout.preferredHeight: metrics.homeTopBarHeight
            placeholderText: "Search"
            onTextChanged: root.searchChanged(text)
        }

        Components.PillButton {
            Layout.preferredWidth: metrics.homeSubscriptionButtonWidth
            Layout.preferredHeight: metrics.homeTopBarHeight
            text: "Subscription"
            variant: "filled"
            iconSource: theme.iconUrl("subscription.svg")
            onClicked: root.subscriptionRequested()
        }

        Components.IconButton {
            Layout.preferredWidth: 62.18
            Layout.preferredHeight: 62.18
            buttonSize: 62.18
            iconSize: 25
            radius: metrics.toolbarButtonRadius
            source: theme.iconUrl("settings_home.svg")
            accessibleLabel: "Settings"
            onClicked: root.settingsRequested()
        }

        Components.IconButton {
            Layout.preferredWidth: 62.18
            Layout.preferredHeight: 62.18
            buttonSize: 62.18
            iconSize: 25
            radius: metrics.toolbarButtonRadius
            source: theme.iconUrl("profile.svg")
            accessibleLabel: "Profile"
            onClicked: root.profileRequested()
        }
    }
}
