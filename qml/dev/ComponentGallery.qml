import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme" as Room36Theme
import "../components" as Components

Item {
    id: root

    property var bridge: null
    readonly property bool hasBridge: bridge !== null && bridge !== undefined

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }

    ScrollView {
        id: scroll
        anchors.fill: parent
        contentWidth: availableWidth

        ColumnLayout {
            width: scroll.availableWidth
            spacing: metrics.spacingLarge

            Item {
                Layout.fillWidth: true
                Layout.preferredHeight: metrics.spacingLarge
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.leftMargin: metrics.spacingLarge
                Layout.rightMargin: metrics.spacingLarge
                spacing: metrics.spacingMedium

                Image {
                    Layout.preferredWidth: 122
                    Layout.preferredHeight: 42
                    source: theme.brandingUrl("logo_blue.svg")
                    fillMode: Image.PreserveAspectFit
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: metrics.spacingTiny

                    Text {
                        Layout.fillWidth: true
                        text: "Component Gallery"
                        color: theme.mainBlue
                        font.family: typography.pageTitleFamily
                        font.pixelSize: typography.pageTitle
                        elide: Text.ElideRight
                    }

                    Text {
                        Layout.fillWidth: true
                        text: typography.hasBrandFonts
                            ? "Britanica loaded: " + typography.thinFamily + ", " + typography.regularFamily + ", " + typography.boldFamily
                            : "Using fallback font: " + typography.fallbackFamily
                        color: theme.mutedText
                        font.family: typography.captionFamily
                        font.pixelSize: typography.caption
                        elide: Text.ElideRight
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.leftMargin: metrics.spacingLarge
                Layout.rightMargin: metrics.spacingLarge
                spacing: metrics.spacingSmall

                Text {
                    text: "Palette"
                    color: theme.black
                    font.family: typography.sectionTitleFamily
                    font.pixelSize: typography.sectionTitle
                }

                Flow {
                    Layout.fillWidth: true
                    Layout.preferredHeight: childrenRect.height
                    spacing: metrics.spacingSmall

                    Column {
                        spacing: metrics.spacingTiny
                        Rectangle { width: 112; height: 56; radius: metrics.radiusMedium; color: theme.mainBlue }
                        Text { text: "mainBlue"; color: theme.black; font.family: typography.captionFamily; font.pixelSize: typography.caption }
                    }

                    Column {
                        spacing: metrics.spacingTiny
                        Rectangle { width: 112; height: 56; radius: metrics.radiusMedium; color: theme.midBlue }
                        Text { text: "midBlue"; color: theme.black; font.family: typography.captionFamily; font.pixelSize: typography.caption }
                    }

                    Column {
                        spacing: metrics.spacingTiny
                        Rectangle { width: 112; height: 56; radius: metrics.radiusMedium; color: theme.lightBlue }
                        Text { text: "lightBlue"; color: theme.black; font.family: typography.captionFamily; font.pixelSize: typography.caption }
                    }

                    Column {
                        spacing: metrics.spacingTiny
                        Rectangle { width: 112; height: 56; radius: metrics.radiusMedium; color: theme.neonGreen }
                        Text { text: "neonGreen"; color: theme.black; font.family: typography.captionFamily; font.pixelSize: typography.caption }
                    }

                    Column {
                        spacing: metrics.spacingTiny
                        Rectangle { width: 112; height: 56; radius: metrics.radiusMedium; color: theme.white; border.width: 1; border.color: theme.separator }
                        Text { text: "white"; color: theme.black; font.family: typography.captionFamily; font.pixelSize: typography.caption }
                    }

                    Column {
                        spacing: metrics.spacingTiny
                        Rectangle { width: 112; height: 56; radius: metrics.radiusMedium; color: theme.black }
                        Text { text: "black"; color: theme.black; font.family: typography.captionFamily; font.pixelSize: typography.caption }
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.leftMargin: metrics.spacingLarge
                Layout.rightMargin: metrics.spacingLarge
                spacing: metrics.spacingSmall

                Text {
                    text: "Typography"
                    color: theme.black
                    font.family: typography.sectionTitleFamily
                    font.pixelSize: typography.sectionTitle
                }

                Text {
                    Layout.fillWidth: true
                    text: "Display / Britanica Thin"
                    color: theme.mainBlue
                    font.family: typography.displayFamily
                    font.pixelSize: typography.display
                    elide: Text.ElideRight
                }

                Text {
                    Layout.fillWidth: true
                    text: "Page title / Britanica Bold"
                    color: theme.black
                    font.family: typography.pageTitleFamily
                    font.pixelSize: typography.pageTitle
                    elide: Text.ElideRight
                }

                Text {
                    Layout.fillWidth: true
                    text: "Body, caption, and detail text use real QML text."
                    color: theme.mutedText
                    font.family: typography.bodyFamily
                    font.pixelSize: typography.body
                    wrapMode: Text.Wrap
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.leftMargin: metrics.spacingLarge
                Layout.rightMargin: metrics.spacingLarge
                spacing: metrics.spacingSmall

                Text {
                    text: "Buttons"
                    color: theme.black
                    font.family: typography.sectionTitleFamily
                    font.pixelSize: typography.sectionTitle
                }

                Flow {
                    Layout.fillWidth: true
                    Layout.preferredHeight: childrenRect.height
                    spacing: metrics.spacingSmall

                    Components.PrimaryButton {
                        text: "Start Scan"
                        iconSource: theme.iconUrl("import.svg")
                    }

                    Components.PrimaryButton {
                        text: "Processing"
                        busy: true
                    }

                    Components.PrimaryButton {
                        text: "Disabled"
                        enabled: false
                    }

                    Components.PillButton {
                        text: "Add"
                        variant: "filled"
                    }

                    Components.PillButton {
                        text: "Disregard"
                        variant: "translucent"
                    }

                    Components.PillButton {
                        text: "Subscription"
                        variant: "accent"
                        iconSource: theme.iconUrl("subscription.svg")
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.leftMargin: metrics.spacingLarge
                Layout.rightMargin: metrics.spacingLarge
                spacing: metrics.spacingSmall

                Text {
                    text: "Toolbar"
                    color: theme.black
                    font.family: typography.sectionTitleFamily
                    font.pixelSize: typography.sectionTitle
                }

                Flow {
                    Layout.fillWidth: true
                    Layout.preferredHeight: childrenRect.height
                    spacing: metrics.spacingSmall

                    Components.IconButton {
                        source: theme.iconUrl("home.svg")
                        accessibleLabel: "Home"
                    }

                    Components.IconButton {
                        source: theme.iconUrl("settings.svg")
                        accessibleLabel: "Settings"
                    }

                    Components.IconButton {
                        source: theme.iconUrl("favorite.svg")
                        accessibleLabel: "Favorite"
                    }

                    Components.SearchField {
                        placeholderText: "Search photos"
                    }

                    Components.ToolbarDropdownButton {
                        label: "Sort"
                        iconSource: theme.iconUrl("sort.svg")
                        options: ["Newest", "Oldest", "File type"]
                    }

                    Components.ToolbarDropdownButton {
                        label: "View"
                        iconSource: theme.iconUrl("view.svg")
                        options: ["Small", "Medium", "Large"]
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.leftMargin: metrics.spacingLarge
                Layout.rightMargin: metrics.spacingLarge
                spacing: metrics.spacingSmall

                Text {
                    text: "Sidebar"
                    color: theme.black
                    font.family: typography.sectionTitleFamily
                    font.pixelSize: typography.sectionTitle
                }

                Flow {
                    Layout.fillWidth: true
                    Layout.preferredHeight: childrenRect.height
                    spacing: metrics.spacingSmall

                    Components.SidebarFilterItem {
                        iconSource: theme.iconUrl("home.svg")
                        label: "Home"
                        count: 36
                        selected: true
                    }

                    Components.SidebarFilterItem {
                        iconSource: theme.iconUrl("landscape.svg")
                        label: "Blurry photos"
                        count: 128
                        showChevron: true
                    }

                    Components.SidebarFilterItem {
                        iconSource: theme.iconUrl("profile.svg")
                        label: "Known people"
                        count: 24
                        expanded: true
                        showChevron: true
                    }

                    Components.SidebarFilterItem {
                        iconSource: theme.iconUrl("trash.svg")
                        label: "Disabled"
                        enabled: false
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.leftMargin: metrics.spacingLarge
                Layout.rightMargin: metrics.spacingLarge
                spacing: metrics.spacingSmall

                Text {
                    text: "Selection And Scores"
                    color: theme.black
                    font.family: typography.sectionTitleFamily
                    font.pixelSize: typography.sectionTitle
                }

                Flow {
                    Layout.fillWidth: true
                    Layout.preferredHeight: childrenRect.height
                    spacing: metrics.spacingMedium

                    Components.SelectionCheckbox {
                        checked: false
                        text: "Unchecked"
                    }

                    Components.SelectionCheckbox {
                        checked: true
                        text: "Selected"
                    }

                    Components.ScoreBar {
                        value: 0.24
                        label: "24%"
                    }

                    Components.ScoreBar {
                        value: 0.67
                        label: "67%"
                    }

                    Components.ScoreBar {
                        value: 0.92
                        label: "92%"
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.leftMargin: metrics.spacingLarge
                Layout.rightMargin: metrics.spacingLarge
                spacing: metrics.spacingSmall

                Text {
                    text: "People"
                    color: theme.black
                    font.family: typography.sectionTitleFamily
                    font.pixelSize: typography.sectionTitle
                }

                Flow {
                    Layout.fillWidth: true
                    Layout.preferredHeight: childrenRect.height
                    spacing: metrics.spacingMedium

                    Components.PersonAvatar {
                        name: "Amina"
                        count: 12
                        showName: true
                        selected: true
                    }

                    Components.PersonAvatar {
                        name: "Noah"
                        count: 8
                        showName: true
                    }

                    Components.PersonAvatar {
                        count: 3
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.leftMargin: metrics.spacingLarge
                Layout.rightMargin: metrics.spacingLarge
                spacing: metrics.spacingSmall

                Text {
                    text: "Cards"
                    color: theme.black
                    font.family: typography.sectionTitleFamily
                    font.pixelSize: typography.sectionTitle
                }

                Flow {
                    Layout.fillWidth: true
                    Layout.preferredHeight: childrenRect.height
                    spacing: metrics.photoCardHorizontalGap

                    Components.PhotoCard {
                        source: theme.backgroundUrl("loading_mesh.svg")
                        filename: "IMG_2042.jpg"
                        selected: true
                        favorite: true
                        badgeText: "Best"
                    }

                    Components.PhotoCard {
                        filename: "DSC_1189.nef"
                        selected: false
                        favorite: false
                        badgeText: "RAW"
                    }

                    Components.ProjectCard {
                        source: theme.backgroundUrl("loading_mesh.svg")
                        title: "Summer Library"
                        subtitle: "248 photos"
                        selected: true
                    }

                    Components.ProjectCard {
                        title: "People Review"
                        photoCount: 64
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.leftMargin: metrics.spacingLarge
                Layout.rightMargin: metrics.spacingLarge
                spacing: metrics.spacingSmall

                Text {
                    text: "Bridge Debug"
                    color: theme.black
                    font.family: typography.sectionTitleFamily
                    font.pixelSize: typography.sectionTitle
                }

                Components.SearchField {
                    id: folderField
                    Layout.fillWidth: true
                    placeholderText: "Folder"
                    text: root.hasBridge ? root.bridge.folder : ""

                    onAccepted: function(value) {
                        if (root.hasBridge) {
                            root.bridge.setFolder(value)
                            root.bridge.refreshFileTypes()
                        }
                    }
                }

                Text {
                    Layout.fillWidth: true
                    text: "Folder: " + (root.hasBridge && root.bridge.folder.length > 0 ? root.bridge.folder : "Not set")
                    color: theme.black
                    font.family: typography.bodyFamily
                    font.pixelSize: typography.body
                    wrapMode: Text.WrapAnywhere
                }

                Text {
                    Layout.fillWidth: true
                    text: "Status: " + (root.hasBridge ? root.bridge.status : "Bridge unavailable")
                    color: theme.black
                    font.family: typography.bodyFamily
                    font.pixelSize: typography.body
                    wrapMode: Text.Wrap
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: metrics.spacingSmall

                    Text {
                        text: root.hasBridge ? root.bridge.countText : "0 items"
                        color: theme.mutedText
                        font.family: typography.captionFamily
                        font.pixelSize: typography.caption
                    }

                    Text {
                        text: root.hasBridge ? root.bridge.elapsedText : "0s"
                        color: theme.mutedText
                        font.family: typography.captionFamily
                        font.pixelSize: typography.caption
                    }
                }

                ProgressBar {
                    Layout.fillWidth: true
                    from: 0
                    to: root.hasBridge ? Math.max(1, root.bridge.progressMaximum) : 1
                    value: root.hasBridge ? root.bridge.progressValue : 0
                    indeterminate: root.hasBridge ? root.bridge.progressIndeterminate : false
                }

                RowLayout {
                    spacing: metrics.spacingSmall

                    Components.PrimaryButton {
                        Layout.preferredWidth: 180
                        Layout.preferredHeight: metrics.toolbarButtonHeight
                        text: "Start Scan"
                        enabled: root.hasBridge && root.bridge.canScan
                        onClicked: root.bridge.startScan()
                    }

                    Components.PillButton {
                        text: "Cancel"
                        variant: "translucent"
                        enabled: root.hasBridge && root.bridge.canCancel
                        onClicked: root.bridge.cancelScan()
                    }
                }
            }

            Item {
                Layout.fillWidth: true
                Layout.preferredHeight: metrics.spacingLarge
            }
        }
    }

    Connections {
        target: root.hasBridge ? root.bridge : null

        function onFolderChanged() {
            if (!folderField.activeFocus) {
                folderField.text = root.bridge.folder
            }
        }
    }
}
